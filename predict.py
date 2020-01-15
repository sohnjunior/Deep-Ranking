import os
import numpy as np
import pandas as pd
from PIL import Image

# from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from utils import euclidean_distance, data_transforms, DatasetImageNet


# -- path info
TRIPLET_PATH = 'triplet.csv'
MODEL_PATH = 'deepranknet.model'
EMBEDDING_PATH = 'embedding.txt'

# -- parameters
BATCH_SIZE = 4

class Prediction:
    def __init__(self):
        self.model = torch.load(MODEL_PATH)
        self.train_df = pd.read_csv(TRIPLET_PATH)
        self.train_embedded = # TODO X_train 에 해당하는 임베딩된 결과들을 담고 있는 변수이다

    def check_embedding(self):
        """ 훈련 이미지들을 임베딩한 결과를 textfile로 저장한다. """
        if not os.path.exist(EMBEDDING_PATH):
            print('==> Generate embedding...')
            model.eval()

            train_dataset = DatasetImageNet(TRAIN_PATH, transform=data_transforms['val'])
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                                      drop_last=True, num_workers=4)

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

    def query_embedding(self, image):
        """ return embedded query image """
        model.eval()
        embedding = model(image)
        return embedding.cpu().detach().numpy()

    def save_result(self, result, result_num, name='result.png'):
        """ save similarity result """
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
        plt.savefig(str(name))  # save as file

    def predict(self, query_image_path, result_num):
        """ predict top-n similar images """
        # 먼저 훈련 이미지 임베딩이 완료되었는지 체크한다.
        self.check_embedding()

        # read query image and pre-processing
        query_image = Image.open(query_image_path).convert('RGB')
        query_image = data_transforms['val'](query_image)

        # embedding query image
        query_embedded = self.query_embedding(query_image)

        #  by euclidean distance, find top ranked similar images
        image_dist = euclidean_distance(self.train_embedded, query_embedded)
        image_dist_indexed = zip(image_dist, range(image_dist.shape[0]))
        image_dist_sorted = sorted(image_dist_indexed, key=lambda x: x[0])

        # top 5 images
        predicted_images = [(img[0], self.train_df.loc[img[1], "query"]) for img in image_dist_sorted[:result_num]]

        # make png file
        self.save_result([(0.0, query_image_path)] + predicted_images, result_num)

def main():
    X_train = X_train[0::4]
    train_df = train_df.loc[0::4, :].reset_index(drop=True)


    # get images for 3 Validation set
    test_classes = [1, 50, 145] # TODO index 가 아니라 경로로 바꾸기
    for i in test_classes:
        ten_imgs(i)


'''
def knn_accuracy():
    y_test = test_df["query"].apply(lambda x: x.split("/")[2])[0:X_test.shape[0]]

    # parse class label
    y_train = train_df["query"].apply(lambda x: x.split("/")[2])[0:X_train.shape[0]]


    knn_model = KNeighborsClassifier(n_neighbors=30, n_jobs=-1, p=2)
    knn_model.fit(X=X_train, y=y_train)
    _, idx = knn_model.kneighbors(X_test, n_neighbors=K)  # idx -> top 30 nearest neighbors
    n_idx = []
    for i in range(0, len(idx)):
        n_idx.append([y_train[x] for x in idx[i]])

    sum([1 if (y_test[i] in n_idx[i]) else 0 for i in range(0, len(idx))]) / float(len(idx))
'''

if __name__ == '__main__':
    main()

