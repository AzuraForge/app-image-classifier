import numpy as np
from typing import Tuple, List
from torchvision import datasets
import os

from azuraforge_learner import Sequential, Conv2D, MaxPool2D, ReLU, Flatten, Linear
from azuraforge_learner.pipelines import ImageClassificationPipeline
class Cifar10Pipeline(ImageClassificationPipeline):
    """CIFAR-10 veri setini kullanarak görüntü sınıflandırma yapar."""

    def _load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """CIFAR-10 verisini torchvision ile indirir ve hazırlar."""
        data_path = os.getenv("CACHE_DIR", ".cache")
        self.logger.info(f"Loading CIFAR-10 data from/to: {data_path}")
        
        train_set = datasets.CIFAR10(root=data_path, train=True, download=True)
        test_set = datasets.CIFAR10(root=data_path, train=False, download=True)

        X_train, y_train = train_set.data, np.array(train_set.targets)
        X_test, y_test = test_set.data, np.array(test_set.targets)
        
        # Veriyi (N, H, W, C) formatından (N, C, H, W) formatına çevir
        X_train = X_train.transpose(0, 3, 1, 2)
        X_test = X_test.transpose(0, 3, 1, 2)
        
        # Veriyi [0, 1] aralığına normalize et
        X_train, X_test = X_train / 255.0, X_test / 255.0

        # Eğitim setinin sadece küçük bir kısmını kullanalım (hızlı olması için)
        train_limit = self.config.get("data_sourcing", {}).get("train_limit", 1000)
        test_limit = self.config.get("data_sourcing", {}).get("test_limit", 200)

        return (
            X_train[:train_limit].astype(np.float32), 
            y_train[:train_limit],
            X_test[:test_limit].astype(np.float32), 
            y_test[:test_limit],
            train_set.classes
        )

    def _create_model(self, input_shape: Tuple, num_classes: int) -> Sequential:
        """Basit bir Konvolüsyonel Sinir Ağı (CNN) oluşturur."""
        self.logger.info("Creating a simple CNN model...")
        
        # Örnek bir CNN Mimarisi
        model = Sequential(
            # Girdi: (N, 3, 32, 32)
            Conv2D(in_channels=3, out_channels=16, kernel_size=3, padding=1), # -> (N, 16, 32, 32)
            ReLU(),
            MaxPool2D(kernel_size=2, stride=2), # -> (N, 16, 16, 16)
            
            Conv2D(in_channels=16, out_channels=32, kernel_size=3, padding=1), # -> (N, 32, 16, 16)
            ReLU(),
            MaxPool2D(kernel_size=2, stride=2), # -> (N, 32, 8, 8)
            
            Flatten(), # -> (N, 32 * 8 * 8) = (N, 2048)
            Linear(32 * 8 * 8, num_classes) # -> (N, 10)
        )
        return model