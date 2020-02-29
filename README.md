Image Similarity with Deep Ranking Algorithm 
============================================ 


Installation
--------------- 
### Required package for Triplet Checker
<pre>
<code>
pip install tkinter
pip install pillow
pip install pandas
pip install ttkthemes
</code>
</pre>

<br>

Usage
--------------- 
### Deep Ranking
<pre>
<code>
python triplet_sampler.py <i>--n_pos</i> 1 <i>--n_neg</i> 1 
python train_net.py <i>--epochs</i> 1 <i>--optim</i> adam 
python predict.py
</code>
</pre>

### Triplet Checker
```
python triplet_checker.py
```
> triplet_checker.py 는 triplet.csv 파일과 같은 디렉토리에 위치시켜야 합니다.
