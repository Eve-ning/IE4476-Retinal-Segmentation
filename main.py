import matplotlib
from sklearn.metrics import recall_score, accuracy_score

from image_io import load_images, save_images, load_other_tests
from retinal_segmenter import RetinalSegmenter

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

if __name__ == '__main__':

    x_train, y_train, x_test = load_images()
    seg = RetinalSegmenter()
    y_pred_train = seg.predict(x_train)

    train_score = seg.score(y_pred_train, y_train)

    print(f"F1 Score: {train_score:.2%}")
    print(f"Sensitivity/Recall: {recall_score(y_pred_train.flatten(), y_train.flatten()):.2%}")
    print(f"Accuracy: {accuracy_score(y_pred_train.flatten(), y_train.flatten()):.2%}")

    save_images({'data/y_pred_test.gif': seg.predict(x_test), 'data/y_pred_train.gif': y_pred_train})

    # Here, we create a collage of the other tests and their results
    x_tests = load_other_tests()
    y_pred_tests = [seg.predict(x) for x in x_tests]
    fig, axs = plt.subplots(10, 6, sharex=True, sharey=True, figsize=(9, 15))
    for ax, y in zip(axs.flatten(), [i for j in zip(x_tests, y_pred_tests) for i in j]):
        ax.axis('off')
        ax.imshow(y, cmap='gray')
    fig.tight_layout()
    fig.savefig("data/other_tests.png")
