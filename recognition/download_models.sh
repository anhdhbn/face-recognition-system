if [ -e ./models/frozen_model.pb ] 
then
    echo "frozen_model.pb downloaded"
else
    echo "downloading frozen_model.pb"
    mkdir -p models && wget frozen_model.pb -P ./models/ http://download.recofat.vn/frozen_model.pb
fi