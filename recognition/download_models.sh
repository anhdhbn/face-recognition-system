if [ -e ./models/frozen_model.pb ] 
then
    echo "frozen_model.pb downloaded"
else
    echo "downloading frozen_model.pb"
    mkdir -p models
    wget frozen_model.zip -P ./models/ http://download.recofat.vn/frozen_model.zip
    unzip ./models/frozen_model.zip -d ./models/
fi