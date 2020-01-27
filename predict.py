import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from net import DeepRank
from utils import euclidean_distance, data_transforms, DatasetImageNet


# -- path info
TRIPLET_PATH = 'triplet.csv'
MODEL_PATH = 'deeprank.pt'
EMBEDDING_PATH = 'embedding.txt'

# -- parameters
BATCH_SIZE = 4


class Prediction:
    def __init__(self):
        self.model = DeepRank()
        self.model.load_state_dict(torch.load(MODEL_PATH))  # load model parameters
        self.train_df = pd.read_csv(TRIPLET_PATH).drop_duplicates('query', keep='first').reset_index(drop=True)

        # check embedding
        if not os.path.exists(EMBEDDING_PATH):
            print('pre-generated [embedding.txt] not exist!')
            self.embedding()
        self.train_embedded = np.fromfile(EMBEDDING_PATH, dtype=np.float32).reshape(-1, 4096)

    def embedding(self):
        """ create embedding textfile with train data """
        print('  ==> Generate embedding...', end='')
        self.model.eval()  # set to eval mode

        train_dataset = DatasetImageNet(TRIPLET_PATH, embedding=True, transform=data_transforms['val'])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                                   shuffle=False, drop_last=True, num_workers=4)

        embedded_images = []
        for batch_idx, (Q, _, _) in enumerate(train_loader):
            if torch.cuda.is_available():
                Q = Variable(Q).cuda()
            else:
                Q = Variable(Q)

            embedding = self.model(Q)
            embedding_np = embedding.cpu().detach().numpy()

            embedded_images.append(embedding_np)  # collect train data's predicted results

        embedded_images_train = np.concatenate(embedded_images, axis=0)
        embedded_images_train.astype('float32').tofile(EMBEDDING_PATH)  # save embedding result
        print('done! [embedding.txt] generated')

    def query_embedding(self, query_image_path):
        """ return embedded query image """
        print(f'Query image [{query_image_path}] embedding...', end='')

        # read query image and pre-processing
        query_image = Image.open(query_image_path).convert('RGB')
        query_image = data_transforms['val'](query_image)
        query_image = query_image[None]  # add new axis. same as 'query_image[None, :, :, :]'

        self.model.eval()  # set to eval mode

        embedding = self.model(query_image)
        print('done!')
        return embedding.cpu().detach().numpy()

    def save_result(self, result, result_num, result_name):
        """ save similarity result """
        print('Save predicted result ...', end='')
        fig = plt.figure(figsize=(64, 64))
        columns = result_num + 1
        ax = []
        for i in range(1, columns + 1):
            dist, img_path = result[i - 1]
            img = mpimg.imread(img_path)  # read image
            ax.append(fig.add_subplot(1, columns, i))
            if i == 1:  # query image
                ax[-1].set_title("query image", fontsize=50)
            else:  # others
                ax[-1].set_title("img_:" + str(i - 1), fontsize=50)
                ax[-1].set(xlabel='l2-dist=' + str(dist))
                ax[-1].xaxis.label.set_fontsize(25)
            plt.imshow(img)
        plt.savefig(result_name)  # save as file
        print('done!')

    def predict(self, query_image_path, result_num, save_as='result.png'):
        """ predict top-n similar images """
        # check query path is valid
        if not os.path.exists(query_image_path):
            print(f'[ERROR] invalid query image path: {query_image_path}')
            return

        # embedding query image
        query_embedded = self.query_embedding(query_image_path)

        #  by euclidean distance, find top ranked similar images
        image_dist = euclidean_distance(self.train_embedded, query_embedded)
        image_dist_indexed = zip(image_dist, range(image_dist.shape[0]))
        image_dist_sorted = sorted(image_dist_indexed, key=lambda x: x[0])

        # top 5 images
        predicted_images = [(img[0], self.train_df.loc[img[1], "query"]) for img in image_dist_sorted[:result_num]]

        # make png file
        self.save_result([(0.0, query_image_path)] + predicted_images, result_num, result_name=save_as)


def main():
    predictor = Prediction()
    image_path1 = 'your validation path'
    image_path2 = 'your validation path'
    image_path3 = 'your validation path'

    # get images for 3 Validation set
    test_images = [image_path1, image_path2, image_path3]
    for idx, p in enumerate(test_images):
        predictor.predict(p, 5, f'result_{idx}.png')


if __name__ == '__main__':
    main()

