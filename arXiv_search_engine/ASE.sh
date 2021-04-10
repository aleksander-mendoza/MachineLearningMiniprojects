# Data from kaggle
# https://www.kaggle.com/Cornell-University/arxiv
#
# 1. unzip
# 2. run jq to see list of IDs
# jq '.id' arxiv-metadata-oai-snapshot.json
# 3. download a couple of PDFs
# wget https://arxiv.org/pdf/{id}
#
# Full procedure:
unzip archive.zip
mkdir pdf
cd pdf
for id in $(jq -r '.id' ../arxiv-metadata-oai-snapshot.json | head -n 100) ; do
    wget --user-agent="Mozilla/5.0 (X11; Fedora; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0" https://arxiv.org/pdf/$id.pdf
done
cd ..

###### Convert to text

for pdf in * ; do
    echo $pdf
    pdftotext $pdf ../txt/${pdf%.*}.txt
done





