import pandas as pd


def get_img_filenames(filename):
    with open(filename) as f:
        img_filenames = f.read().splitlines()
    return img_filenames

train_images = get_img_filenames("Flickr_8k.trainImages.txt")
validation_images = get_img_filenames("Flickr_8k.devImages.txt")
test_images = get_img_filenames("Flickr_8k.testImages.txt")

assert (len(train_images), len(validation_images), len(test_images)) == (6000, 1000, 1000)

df_groups = pd.concat([
    pd.DataFrame({"data_group": "training", "filename": train_images}),
    pd.DataFrame({"data_group": "validation", "filename": validation_images}),
    pd.DataFrame({"data_group": "test", "filename": test_images})
])

df_captions = (
    pd.read_table("Flickr8k.token.txt", header=None)
    .rename(columns={0: "filename", 1: "caption"})
    # Remove #1-#5 from filenames
    .assign(filename = lambda x: x["filename"].replace('#\d', '', regex=True))
    # Remove period and trailing whitespace
    .assign(caption = lambda x: x["caption"].str.replace('.', '').str.strip())
    # Wrap captions inside of <caption>...</caption>
    .assign(caption = lambda x: '<caption>' + x['caption'] + '</caption>')
    # Add data groups
    .merge(df_groups, on="filename", how="left")
    # Missing data groups are added to training
    .assign(data_group = lambda x: x["data_group"].fillna("training"))
)

df_captions.to_csv("munged_captions.tsv.gz", sep="\t", compression="gzip")