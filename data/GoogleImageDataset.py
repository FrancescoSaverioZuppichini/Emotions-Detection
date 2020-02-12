from torch.utils.data import Dataset
from google_images_download import google_images_download
from pathlib import Path
from torchvision.datasets.folder import ImageFolder


class GoogleImageDataset(ImageFolder):
    @classmethod
    def from_google(cls, keywords, output_directory, *args, **kwargs):
        out_dir = Path(output_directory)
        out_dir.mkdir(exist_ok=True)
        response = google_images_download.googleimagesdownload()
        absolute_image_paths = response.download({
            "keywords": keywords,
            "output_directory": output_directory,
            "image_directory": keywords,
            **kwargs
        })
        return cls(root=output_directory)


GoogleImageDataset.from_google(
    keywords="Smile",
    output_directory='./dataset/',
    limit=20,
    print_urls=True
)
