from transforms import *
from utils import *
from dataset import *
from model import *
from losses import *

logdir = "../logs/segmentation/" + NETWORK + "_" + ENCODER
paramsdir = "../parsed_data/optimal_thresholds"
os.makedirs(paramsdir, exist_ok=True)

def main():
    model, dataset, loaders = initialize_model(NETWORK, 'infer')

    runner = SupervisedRunner(device=utils.get_device())
    callbacks = [
        CheckpointCallback(resume=f"{logdir}/checkpoints/best.pth"),
        InferCallback()
    ]
    runner.infer(
        model = model,
        loaders = loaders,
        callbacks = callbacks
    )
    print("Inference made using optimal model...")

    # GET ALL 4 CLASS MASKS FOR EACH VALIDATION IMAGE
    valid_masks = []
    probabilities = np.zeros((2220, 350, 525))
    for i, (batch, output) in enumerate(tqdm(zip(dataset['valid'], runner.callbacks[0].predictions["logits"]))):
        image, mask = batch
        for m in mask:
            if m.shape != (350, 525):
                m = cv2.resize(m, dsize=(525, 350), interpolation=cv2.INTER_LINEAR)
            valid_masks.append(m)

        for j, probability in enumerate(output):
            if probability.shape != (350, 525):
                probability = cv2.resize(probability, dsize=(525, 350), interpolation=cv2.INTER_LINEAR)
            probabilities[i * 4 + j, :, :] = probability

    # USE VALIDATION DATA TO FIND BEST THRESHOLDS FOR EACH CLASS
    class_params = {}
    for class_id in range(4):
        print(class_id)
        attempts = []
        for t in range(0, 100, 5):
            t /= 100
            for ms in [0, 100, 1200, 5000, 10000]:
                masks = []
                for i in range(class_id, len(probabilities), 4):
                    probability = probabilities[i]
                    predict, num_predict = post_process(sigmoid(probability), t, ms)
                    masks.append(predict)

                d = []
                for i, j in zip(masks, valid_masks[class_id::4]):
                    if (i.sum() == 0) & (j.sum() == 0):
                        d.append(1)
                    else:
                        d.append(dice(i, j))

                attempts.append((t, ms, np.mean(d)))

        attempts_df = pd.DataFrame(attempts, columns=['threshold', 'size', 'dice'])

        attempts_df = attempts_df.sort_values('dice', ascending=False)
        print(attempts_df.head())
        best_threshold = attempts_df['threshold'].values[0]
        best_size = attempts_df['size'].values[0]

        class_params[class_id] = (best_threshold, best_size)


    # Save Optimal Params to File
    outfile = open(paramsdir+"/optimal_thresholds_"+NETWORK+"_"+ENCODER, 'wb')
    pickle.dump(class_params, outfile)
    outfile.close()
    print(class_params)

    # FREE GPU CACHE
    del model, dataset, loaders, runner
    torch.cuda.empty_cache()
    gc.collect()

    # visualize Threshold and min size vs dice for one of the classes
    #sns.lineplot(x='threshold', y='dice', hue='size', data=attempts_df);
    #plt.title('Threshold and min size vs dice for one of the classes');


def visualize_masks():
    for i, (input, output) in enumerate(zip(
            valid_dataset, runner.callbacks[0].predictions["logits"])):
        image, mask = input

        image_vis = image.transpose(1, 2, 0)
        mask = mask.astype('uint8').transpose(1, 2, 0)
        pr_mask = np.zeros((350, 525, 4))
        for j in range(4):
            probability = cv2.resize(output.transpose(1, 2, 0)[:, :, j], dsize=(525, 350), interpolation=cv2.INTER_LINEAR)
            pr_mask[:, :, j], _ = post_process(sigmoid(probability), class_params[j][0], class_params[j][1])
        #pr_mask = (sigmoid(output) > best_threshold).astype('uint8').transpose(1, 2, 0)

        visualize_with_raw(image=image_vis, mask=pr_mask, original_image=image_vis, original_mask=mask, raw_image=image_vis, raw_mask=output.transpose(1, 2, 0))

        if i >= 2:
            break



if __name__ == '__main__':
    main()
