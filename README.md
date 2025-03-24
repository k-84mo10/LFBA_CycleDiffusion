# LFBA_CycleDiffusion

## 環境構築
まず、data-raidから学習させたいデータのシンボリックリンクを貼る。
```sh
ln -s /mnt/data-raid/hashimoto/dx/data/dataset_live/dataset_live_after_processed_splited data
ln -s /mnt/data-raid/hashimoto/dx/data/back_image_2 back_data
```
次に仮想環境を作る。
```sh
python3 -m venv myenv
source myenv/bin/activate
```
そして、ライブラリをインストールする。
```sh
pip install -r requirements_pytorch.txt --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements_others.txt
```