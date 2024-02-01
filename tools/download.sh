mkdir -p data/scienceqa/images
cd data/scienceqa/images

aria2c -c -x5 https://scienceqa.s3.us-west-1.amazonaws.com/images/train.zip
aria2c -c -x5 https://scienceqa.s3.us-west-1.amazonaws.com/images/val.zip
aria2c -c -x5 https://scienceqa.s3.us-west-1.amazonaws.com/images/test.zip

unzip -q train.zip
unzip -q val.zip
unzip -q test.zip

