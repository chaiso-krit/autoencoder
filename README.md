# autoencoder
Autoencoder using Theano library, following the [UFLDL Tutorial](http://ufldl.stanford.edu/wiki/index.php/Exercise:Sparse_Autoencoder),Stanford University

## Prerequire:
  - python-numpy python-scipy python-pip python-matplotlib
  
  ```sh
  $ sudo apt-get install python-numpy python-scipy python-pip python-matplotlib
  ```
  - theano Library
  
  ```sh
  $ sudo pip install theano
  ```

## How To Use:
  - Using CPU 
  
  ```sh
  $ python sparse_autoencoder.py
  ```
  - Using GPU
  
  ```sh
  $ THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python sparse_autoencoder.py
  ```
