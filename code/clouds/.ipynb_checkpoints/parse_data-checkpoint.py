from utils import *

def main():
    data_folders = os.listdir(path)

    # get TRAIN.CSV and SAMPLE SUBMISSION.CSV
    train = pd.read_csv(f'{path}/train.csv')
    sub = pd.read_csv(f'{path}/sample_submission.csv')

    # get NUMBER OF TRAINING/TEST DATA POINTS
    n_train = len(os.listdir(f'{path}/train_images'))
    n_test = len(os.listdir(f'{path}/test_images'))
    print(f'There are {n_train} images in train dataset')
    print(f'There are {n_test} images in test dataset')

    # CREATE LABEL and IMAGE ID FIELDS FOR TRAIN.CSV
    train['label'] = train['Image_Label'].apply(lambda x: x.split('_')[1])
    train['im_id'] = train['Image_Label'].apply(lambda x: x.split('_')[0])
    sub['label'] = sub['Image_Label'].apply(lambda x: x.split('_')[1])
    sub['im_id'] = sub['Image_Label'].apply(lambda x: x.split('_')[0])

    # PREPARE DATA FOR MODELLING
    # create a list of unique image ids and the count of masks for images.
    # This will allow us to make a stratified split based on this count.
    id_mask_count = train.loc[train['EncodedPixels'].isnull() == False, 'Image_Label']\
        .apply(lambda x: x.split('_')[0])\
        .value_counts()\
        .reset_index()\
        .rename(columns={'index': 'img_id', 'Image_Label': 'count'})

    train_ids, valid_ids = train_test_split(
        id_mask_count['img_id'].values,
        random_state=42,
        stratify=id_mask_count['count'],
        test_size=0.1)

    test_ids = sub['Image_Label']\
        .apply(lambda x: x.split('_')[0])\
        .drop_duplicates().values

    # SAVE PARSED DATA TO FILE
    os.makedirs('../parsed_data/', exist_ok=True)
    train.to_csv('../parsed_data/train.csv')
    sub.to_csv('../parsed_data/submission.csv')
    #id_mask_count.to_csv(path+'/parsed_data/id_mask_count.csv')

    outfile_train = open("../parsed_data/train_ids", 'wb')
    outfile_valid = open("../parsed_data/valid_ids", 'wb')
    outfile_test = open("../parsed_data/test_ids", 'wb')
    pickle.dump(train_ids, outfile_train)
    pickle.dump(valid_ids, outfile_valid)
    pickle.dump(test_ids, outfile_test)
    outfile_train.close()
    outfile_valid.close()
    outfile_test.close()




if __name__ == '__main__':
    main()
