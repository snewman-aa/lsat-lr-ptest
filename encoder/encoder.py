import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.utils import murmurhash3_32
from scipy.linalg import orth
from scipy.fft import fft, ifft


class Encoder:
    def __init__(self, output_dim=10000, emb_model='sbert'):
        """
        Initialize the Encoder.
        :param output_dim: Target dimensionality for the HDVs.
        :param emb_model: Which embedding model to use; e.g., 'sbert', 'gemini', or 'mistral'.
        """
        self.output_dim = output_dim
        self.emb_model = emb_model
        self.model = self._init_embedding_model(emb_model)

    def _init_embedding_model(self, emb_model):
        """
        Initialize the sentence embedding model.
        """
        if emb_model == 'sbert':
            return SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        elif emb_model == 'gemini':
            raise NotImplementedError("Gemini model not implemented.")
        elif emb_model == 'mistral':
            return SentenceTransformer('Linq-AI-Research/Linq-Embed-Mistral', device='cpu')
        else:
            raise ValueError(f"Unknown embedding model: {emb_model}")

    def generate_orthogonal_roles(self, num_roles=3):
        """
        Generate orthogonal role vectors using Gram–Schmidt via scipy.linalg.orth.
        :param num_roles: Number of roles (e.g., 3 for 'stimulus', 'prompt', and 'explanation').
        :return: Dictionary mapping field names to orthogonal role vectors.
        """
        random_matrix = np.random.randn(self.output_dim, num_roles)
        ortho_matrix = orth(random_matrix)
        ortho_matrix = ortho_matrix[:, :num_roles]
        roles = {
            "stimulus": ortho_matrix[:, 0],
            "prompt": ortho_matrix[:, 1],
            "explanation": ortho_matrix[:, 2],
        }
        return roles

    def get_sentence_embeddings(self, sentences: list) -> np.ndarray:
        """
        Generate sentence embeddings for a list of sentences.
        :param sentences: List of strings.
        :return: Array of embeddings.
        """
        if self.emb_model in ['sbert', 'mistral']:
            embeddings = self.model.encode(sentences)
        elif self.emb_model == 'gemini':
            # Example: return self.model.embed(sentences)
            raise NotImplementedError("Gemini model not implemented in this example.")
        else:
            embeddings = np.zeros((len(sentences), 384))
        return embeddings

    def simhash_projection(self, embedding: np.ndarray) -> np.ndarray:
        """
        Project a dense embedding to a high-dimensional real-valued vector (HDV)
        using a hash-based accumulation.
        :param embedding: Input embedding (1D numpy array).
        :return: Real-valued HDV (1D numpy array of length output_dim).
        """
        real_hdv = np.zeros(self.output_dim)
        for i, value in enumerate(embedding):
            hash_index = murmurhash3_32(i, positive=True) % self.output_dim
            real_hdv[hash_index] += value
        return real_hdv

    def project_to_hdvs(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Project a list/array of embeddings to high-dimensional vectors.
        :param embeddings: 2D numpy array of sentence embeddings.
        :return: Array of HDVs.
        """
        hdvs = np.array([self.simhash_projection(emb) for emb in embeddings])
        return hdvs

    def circular_convolve(self, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        """
        Perform circular convolution of two vectors using FFT.
        :param v1: First vector.
        :param v2: Second vector.
        :return: Circularly convolved vector.
        """
        return np.real(ifft(fft(v1) * fft(v2)))

    def bind(self, role_hdv: np.ndarray, value_hdv: np.ndarray) -> np.ndarray:
        """
        Bind a role vector to a value vector via circular convolution.
        :param role_hdv: Role vector.
        :param value_hdv: Value vector.
        :return: Bound HDV.
        """
        return self.circular_convolve(role_hdv, value_hdv)

    def bundle(self, bound_hdvs: list) -> np.ndarray:
        """
        Bundle multiple HDVs by element-wise addition.
        :param bound_hdvs: List of HDV vectors.
        :return: Bundled HDV.
        """
        bundled_hdv = np.sum(bound_hdvs, axis=0)
        return bundled_hdv

    def get_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Compute a cosine similarity matrix for a set of embeddings.
        :param embeddings: 2D numpy array where each row is an embedding.
        :return: 2D cosine similarity matrix.
        """
        sim_matrix = np.dot(embeddings, embeddings.T)
        norms = np.linalg.norm(embeddings, axis=1)
        sim_matrix /= np.outer(norms, norms)
        return sim_matrix

    def generate_question_hdv_from_json(
            self,
            json_obj: dict,
            roles: dict,
            weights: dict[str, float] = None
    ) -> np.ndarray:
        """
        Generate an HDV representing an LSAT question by binding field roles to their values
        and bundling the results.
        :param json_obj: A dictionary representing the question
        :param roles: A dictionary of role vectors (e.g., from generate_orthogonal_roles())
        :param weights: optional {"stimulus":w1, "prompt":w2, "explanation":w3}
                        defaults to all 1.0
        :return: A bundled HDV (1D numpy array) representing the question.
        """
        field_values = {
            "stimulus": json_obj.get("stimulus", ""),
            "prompt": json_obj.get("prompt", ""),
            "explanation": json_obj.get("explanation", ""),
        }
        print(f"Generating HDV for question {json_obj.get('question_number', 'unknown')}")
        sentences = list(field_values.values())
        value_embeddings = self.get_sentence_embeddings(sentences)
        value_hdvs = self.project_to_hdvs(value_embeddings)
        bound_hdvs = [self.bind(roles[key], value_hdvs[i]) for i, key in enumerate(field_values.keys())]
        question_hdv = self.bundle(bound_hdvs)
        return question_hdv


if __name__ == '__main__':
    encoder = Encoder(output_dim=10000, emb_model='sbert')
    roles = encoder.generate_orthogonal_roles(num_roles=3)
    # example JSON object for a question:
    sample_json = {
        "question_number": 1,
        "stimulus": "Example stimulus text.",
        "prompt": "What is the main point?",
        "explanation": "Explanation goes here."
    }
    hdv = encoder.generate_question_hdv_from_json(sample_json, roles)
    print("HDV shape:", hdv.shape)
