if [ -e ./data/data.pkl ] 
then
    echo "data.pkl downloaded"
else
    echo "downloading data.pkl"
    mkdir -p data && wget data.pkl -P ./data/ http://download.recofat.vn/data.pkl
fi