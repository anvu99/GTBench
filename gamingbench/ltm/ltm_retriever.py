import numpy as np

try:
    import faiss
except ImportError:
    faiss = None

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm))

class LTMRetriever:
    """
    Handles similarity search to retrieve relevant LTM signals for a given board state.
    Uses FAISS (Facebook AI Similarity Search) to quickly compute cosine similarity 
    between the current board embedding and all stored memory centroids.
    """
    def __init__(
        self,
        signals: list,
        mode: str,
        top_k: int = 1,
        threshold: float = 0.6,
        max_results: int = 3,
        embedder=None,
        type_filter: str = None
    ):
        if faiss is None:
            raise ImportError("faiss is not installed. Please install faiss-cpu or faiss-gpu.")
            
        self.mode = mode
        self.top_k = top_k
        self.threshold = threshold
        self.max_results = max_results
        self.embedder = embedder
        self.type_filter = type_filter
        
        # Pre-filter by type if requested
        if self.type_filter:
            type_str = f"Type: {self.type_filter}"
            self.signals = [s for s in signals if type_str in s.get("text", "")]
        else:
            self.signals = signals

        # "Cold Start" signals are generic or new signals that don't have board state anchors (centroids) yet.
        # We always retrieve these by default to ensure they are available to the agent.
        self.cold_start = [s for s in self.signals if not s.get("centroids")]
        self.anchored_signals = [s for s in self.signals if s.get("centroids")]
        
        # Build FAISS index for fast centroid matching
        # Using IndexFlatIP for inner product (which equals cosine similarity if vectors are L2-normalized)
        self.index = None
        self.index_to_signal = {}
        
        if self.anchored_signals:
            dimension = len(self.anchored_signals[0]["centroids"][0]["vec"])
            self.index = faiss.IndexFlatIP(dimension)
            
            vecs_to_add = []
            idx = 0
            for sig in self.anchored_signals:
                for c in sig["centroids"]:
                    vec = np.array(c["vec"], dtype=np.float32)
                    norm = np.linalg.norm(vec)
                    if norm > 0:
                        vec = vec / norm
                    vecs_to_add.append(vec)
                    self.index_to_signal[idx] = sig
                    idx += 1
                    
            if vecs_to_add:
                vecs_np = np.vstack(vecs_to_add)
                self.index.add(vecs_np)

    def retrieve(self, query_board: str) -> list:
        results = []
        # Always include cold-start signals
        for s in self.cold_start:
            results.append({"signal": s, "score": 1.0})
            
        if self.index and self.embedder:
            # The query_board is the current game state, so it gets the instruction prefix
            query_vec = self.embedder.encode(query_board, is_query=True)
            query_vec = np.array(query_vec, dtype=np.float32)
            norm = np.linalg.norm(query_vec)
            if norm > 0:
                query_vec = query_vec / norm
            query_vec = query_vec.reshape(1, -1)
            
            # Retrieve a sufficiently large number of centroids to deduplicate
            # We use min(k, total_elements) to avoid warnings
            k_search = min(self.max_results, self.index.ntotal) if self.max_results > 0 else self.index.ntotal
            print(f"DEBUG RETRIEVAL: k_search={k_search}, ntotal={self.index.ntotal}, query_vec_shape={query_vec.shape}")
            scores, indices = self.index.search(query_vec, k_search)
            print(f"DEBUG RETRIEVAL: scores={scores}, indices={indices}")
            
            seen_names = set()
            for score, idx in zip(scores[0], indices[0]):
                print(f"DEBUG RETRIEVAL: score={score}, idx={idx}")
                if idx == -1:
                    continue
                sig = self.index_to_signal[idx]
                if sig["name"] not in seen_names:
                    seen_names.add(sig["name"])
                    
                    if self.mode == "threshold":
                        if score >= self.threshold:
                            results.append({"signal": sig, "score": float(score)})
                    else:  # top_k
                        results.append({"signal": sig, "score": float(score)})
                        
        # Sort retrieved centroids by score descending
        results.sort(key=lambda x: x["score"], reverse=True)
        
        # Deduplicate results: multiple centroids might map to the same underlying signal,
        # but we only want to inject the signal text once into the LLM context.
        # We also deduplicate against cold_start signals.
        final_results = []
        final_names = set()
        for r in results:
            if r["signal"]["name"] not in final_names:
                final_results.append(r["signal"])
                final_names.add(r["signal"]["name"])
        
        print(f"DEBUG RETRIEVAL: final_results length={len(final_results)}")
        # Apply caps based on mode
        if self.mode == "top_k":
            final_results = final_results[:self.top_k]
        elif self.mode == "threshold":
            final_results = final_results[:self.max_results]
            
        # Process dynamic examples for retrieved signals
        # If a retrieved signal has experiential examples, we find the one closest to the 
        # current board state and inject it dynamically into the prompt text as a 'Historical Record'.
        output_signals = []
        for sig in final_results:
            sig_copy = sig.copy()
            if sig.get("examples") and self.embedder:
                best_ex = self._find_best_example(sig["examples"], query_board)
                if best_ex:
                    safe_past_board = best_ex['board'].replace('--- ONGOING CHAT ---', '--- PAST CHAT HISTORY ---')
                    example_text = f"\n\n  --------------------------------------------------------------------------------\n  [Historical Record] In a highly similar past situation, you successfully applied this strategy by taking the following action:\n\n  Past Board State:\n{safe_past_board}\n\n  Your Past Action: {best_ex['action']}\n  --------------------------------------------------------------------------------"
                    sig_copy["text"] = sig_copy.get("text", "") + example_text
            output_signals.append(sig_copy)
            
        return output_signals

    def _find_best_example(self, examples: list, query_board: str) -> dict:
        """
        Calculates cosine similarity between the current query_board and the board states 
        of all stored examples for a signal, returning the single most relevant example.
        """
        if not examples:
            return None
            
        # We use is_query=True for the query board to match FAISS retrieval
        query_vec = self.embedder.encode(query_board, is_query=True)
        best_sim = -1.0
        best_ex = None
        
        for ex in examples:
            # Examples are stored documents, so is_query=False (default)
            ex_vec = self.embedder.encode(ex["board"], is_query=False)
            sim = cosine_sim(ex_vec, query_vec)
            if sim > best_sim:
                best_sim = sim
                best_ex = ex
                
        return best_ex
