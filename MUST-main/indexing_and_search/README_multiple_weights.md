Step 1: Download celeba dataset
Go to https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8?resourcekey=0-5BR16BdXnb8hVj6CNHKzLg and download folder Anno, Eval, Img/img_align_celeba_png.7z (or Img/imag_align_celebra.zip)

move Anno/list_attr_celeba.txt, Eval/list_eval_partition.txt, and Img/img_align_celeba to a folder under MUST-main (e.g., celeba)

Step 2: transform dataset into vectors
go to MUST-main/indexing_and_search/scripts, run the command below to transform the "train" dataset into vectors with images as modal1 and calcuate ground truth.
```
python3 prepare_celeba_resnet50.py --celeba-root ../../celeba --out-root ../doc/dataset/celeba --type train --base-split 0 --query-split 0 --gt-topk 10 --gt-mode modal1 --seed 42
```

If finding ground truth takes too much time, you can restrict the query size to N (e.g., 100)
```
python3 prepare_celeba_resnet50.py --celeba-root --celeba-root ../../celeba --out-root ../doc/dataset/celeba --type train --base-split 0 --query-split 0 --max-query 100 --gt-topk 10 --gt-mode modal1 --seed 42
```

If resources are limited, try use smaller dataset "test"
```
python3 prepare_celeba_resnet50.py --celeba-root ../../celeba --out-root ../doc/dataset/celeba --type test --base-split 0 --query-split 0 --max-query 100 --gt-topk 10 --gt-mode modal1 --seed 42
```

Step 3: generate graph index with given weights
