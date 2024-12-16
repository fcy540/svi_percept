from svi_percept import SVIPerceptPipeline
from torch.utils.data import Dataset
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        return {'image': image}  # Return the actual PIL Image, not the path

def dataset_example(pipeline):
    # Create dataset from image paths
    image_paths = ['image1.jpg', 'image2.jpg']
    dataset = ImageDataset(image_paths)

    # Use pipeline with dataset
    results = pipeline(dataset)
    print(list(results))

if __name__=='__main__':
    print('simple pipeline')
    pipeline = SVIPerceptPipeline()

    print('test 1')
    print(pipeline('image1.jpg'))

    print('pipeline with batch_size 32')
    pipeline = SVIPerceptPipeline(batch_size=32)

    print('test 2')
    print(pipeline(['image1.jpg', 'image2.jpg']))

    print('test 3')
    print(pipeline([{'image': 'image1.jpg'}, {'image': 'image2.jpg'}]))

    print('test 4')
    def loadimg(fn): return Image.open(fn).convert('RGB')
    print(pipeline([{'image': loadimg('image1.jpg')}, {'image': loadimg('image2.jpg')}]))

    print('test 5')
    dataset_example(pipeline)
