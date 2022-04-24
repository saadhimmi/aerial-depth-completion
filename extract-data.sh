
DIRECTORY=/cluster/scratch/shimmi/aerial-depth-completion/full-ircheldata/
AWS=/cluster/home/shimmi/aws/install-dir/aws-cli/v2/2.4.7/bin/aws
echo $AWS 

# $AWS s3 cp s3://v4rlbagfiles/tarfiles $DIRECTORY --recursive
# tar -xvf $DIRECTORY/*.tar -C $DIRECTORY/sequences
for i in $DIRECTORY/*.tar; do tar -xvf "$i" -C $DIRECTORY/sequences;done

# AWS s3api get-object --bucket v4rlbagfiles --key $FILENAME.tar $DIRECTORY/$FILENAME.tar
# tar -xvf $DIRECTORY/$FILENAME.tar -C $DIRECTORY
