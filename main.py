import sys
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import recall_score, accuracy_score

from image_io import load_images, save_images, load_other_tests
from retinal_segmenter import RetinalSegmenter

if __name__ == '__main__':

    output_folder = Path(sys.argv[-1] if len(sys.argv) > 1 else "preds")
    output_folder.mkdir(parents=True, exist_ok=True)

    print(f"Prediction Outputs at {output_folder}/ directory")

    TRAIN_PREDICT_PATH = f'{output_folder}/y_pred_train.tif'
    TEST_PREDICT_PATH = f'{output_folder}/y_pred_test.tif'
    OTHER_TEST_PREDICT_PATH = f'{output_folder}/other_tests.png'

    x_train, y_train, x_test = load_images()
    seg = RetinalSegmenter()
    y_pred_train = seg.predict(x_train)

    train_score = seg.score(y_pred_train, y_train)

    print("Evaluating Train Accuracy of Training Sample ...")

    print("======== Statistics ========")

    print(f"F1 Score: {train_score:.2%}")
    print(f"Sensitivity/Recall: {recall_score(y_pred_train.flatten(), y_train.flatten()):.2%}")
    print(f"Accuracy: {accuracy_score(y_pred_train.flatten(), y_train.flatten()):.2%}")

    print("============================")

    print(f"Saving Prediction of Training Sample in {TRAIN_PREDICT_PATH} ...")
    print(f"Saving Prediction of Testing Sample in {TEST_PREDICT_PATH} ...")

    save_images({TRAIN_PREDICT_PATH: seg.predict(x_test),
                 TEST_PREDICT_PATH: y_pred_train})

    print("Predicting other test samples as a collage")
    # Here, we create a collage of the other tests and their results
    x_tests = load_other_tests()
    y_pred_tests = [seg.predict(x) for x in x_tests]
    fig, axs = plt.subplots(10, 6, sharex=True, sharey=True, figsize=(9, 15))
    for ax, y in zip(axs.flatten(), [i for j in zip(x_tests, y_pred_tests) for i in j]):
        ax.axis('off')
        ax.imshow(y, cmap='gray')

    print(f"Saving Prediction of Other Test Samples in {OTHER_TEST_PREDICT_PATH} ...")
    fig.tight_layout()
    fig.savefig(OTHER_TEST_PREDICT_PATH)
