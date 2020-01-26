from utils import *
from dataset import *
from transforms import *
from model import *

logdir = "../logs/segmentation/" + NETWORK + "_" + ENCODER
paramsdir = "../parsed_data/optimal_thresholds"
resdir = "../submissions/" + NETWORK + "_" + ENCODER
os.makedirs(resdir, exist_ok=True)

def main():
    torch.cuda.empty_cache()
    gc.collect()

    # Get <submission.csv> Table
    sub = pd.read_csv('../parsed_data/submission.csv')

    # Get Best Class Params
    infile = open(paramsdir+"/optimal_thresholds_"+NETWORK+"_"+ENCODER, 'rb')
    class_params = pickle.load(infile)
    infile.close()

    # Get Best Learned Model from Logs
    best_model_path = f"{logdir}/checkpoints/best.pth"
    model = load_model(NETWORK, best_model_path)

    # Get Loaders
    _, _, loaders = initialize_model(NETWORK, 'test')

    # Load Runner with Best Learned Model
    runner = SupervisedRunner(
        model=model,
        device=utils.get_device()
    )

    encoded_pixels = []
    image_id = 0
    for i, test_batch in enumerate(tqdm(loaders['test'])):

        runner_out = runner.predict_batch(
            {"features": test_batch[0].cuda()}
        )['logits']
        for i, batch in enumerate(runner_out):
            for probability in batch:

                probability = probability.cpu().detach().numpy()
                if probability.shape != (350, 525):
                    probability = cv2.resize(probability, dsize=(525, 350), interpolation=cv2.INTER_LINEAR)
                predict, num_predict = post_process(sigmoid(probability), class_params[image_id % 4][0], class_params[image_id % 4][1])
                if num_predict == 0:
                    encoded_pixels.append('')
                else:
                    r = mask2rle(predict)
                    encoded_pixels.append(r)
                image_id += 1

    sub['EncodedPixels'] = encoded_pixels
    sub.to_csv(resdir + '/submission.csv', columns=['Image_Label', 'EncodedPixels'], index=False)


if __name__ == '__main__':
    main()
