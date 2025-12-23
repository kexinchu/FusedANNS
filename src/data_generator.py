"""
Data generation utilities for motivation tests
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
import random


class MultiModalDataGenerator:
    """Generate synthetic multi-modal data for testing"""
    
    def __init__(self, dim: int = 128, seed: int = 42):
        self.dim = dim
        self.rng = np.random.RandomState(seed)
        random.seed(seed)
    
    def generate_object(self, num_text_vectors: int = 1, 
                       num_image_vectors: int = 1,
                       num_video_vectors: int = 0) -> Dict[str, List[np.ndarray]]:
        """
        Generate a multi-modal object with multiple vectors per modality
        """
        obj = {}
        
        if num_text_vectors > 0:
            obj['text'] = [self.rng.randn(self.dim).astype(np.float32) 
                          for _ in range(num_text_vectors)]
            # Normalize
            for vec in obj['text']:
                vec /= np.linalg.norm(vec) + 1e-8
        
        if num_image_vectors > 0:
            obj['image'] = [self.rng.randn(self.dim).astype(np.float32)
                           for _ in range(num_image_vectors)]
            for vec in obj['image']:
                vec /= np.linalg.norm(vec) + 1e-8
        
        if num_video_vectors > 0:
            obj['video'] = [self.rng.randn(self.dim).astype(np.float32)
                           for _ in range(num_video_vectors)]
            for vec in obj['video']:
                vec /= np.linalg.norm(vec) + 1e-8
        
        return obj
    
    def generate_query(self, modalities: List[str] = ['text']) -> Dict[str, np.ndarray]:
        """Generate a query with specified modalities"""
        query = {}
        for mod in modalities:
            vec = self.rng.randn(self.dim).astype(np.float32)
            vec /= np.linalg.norm(vec) + 1e-8
            query[mod] = vec
        return query
    
    def generate_dataset(self, num_objects: int = 1000,
                        multi_vector_config: Optional[Dict[str, int]] = None) -> List[Dict]:
        """
        Generate a dataset of multi-modal objects
        
        Args:
            num_objects: Number of objects to generate
            multi_vector_config: Dict like {'text': 3, 'image': 5} for multi-vector
        """
        if multi_vector_config is None:
            multi_vector_config = {'text': 1, 'image': 1}
        
        dataset = []
        for i in range(num_objects):
            obj = self.generate_object(
                num_text_vectors=multi_vector_config.get('text', 1),
                num_image_vectors=multi_vector_config.get('image', 1),
                num_video_vectors=multi_vector_config.get('video', 0)
            )
            dataset.append(obj)
        
        return dataset
    
    def generate_ground_truth(self, queries: List[Dict], 
                              objects: List[Dict],
                              relevance_func) -> List[List[float]]:
        """
        Generate ground truth relevance scores
        """
        ground_truth = []
        for query in queries:
            scores = []
            for obj in objects:
                score = relevance_func(query, obj)
                scores.append(score)
            ground_truth.append(scores)
        return ground_truth


def create_multi_caption_object(base_vector: np.ndarray, 
                                num_captions: int = 3,
                                noise_level: float = 0.1) -> List[np.ndarray]:
    """Create multiple caption vectors from a base vector"""
    captions = []
    for _ in range(num_captions):
        noise = np.random.randn(*base_vector.shape).astype(np.float32) * noise_level
        caption = base_vector + noise
        caption /= np.linalg.norm(caption) + 1e-8
        captions.append(caption)
    return captions


def create_multi_frame_video(base_vector: np.ndarray,
                            num_frames: int = 10,
                            noise_level: float = 0.05) -> List[np.ndarray]:
    """Create multiple frame vectors from a base vector"""
    frames = []
    for _ in range(num_frames):
        noise = np.random.randn(*base_vector.shape).astype(np.float32) * noise_level
        frame = base_vector + noise
        frame /= np.linalg.norm(frame) + 1e-8
        frames.append(frame)
    return frames

