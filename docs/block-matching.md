# Block Matching

- [Block Matching](#block-matching)
  - [Algorithm](#algorithm)
  - [Parameters](#parameters)
    - [Window Size](#window-size)
    - [Search Range](#search-range)
    - [Method](#method)
  - [Results](#results)
    - [First Image](#first-image)
    - [Second Image](#second-image)
    - [Third Image](#third-image)

Block matching is a technique used in stereo vision to find the disparity between two images. The disparity is the difference in the horizontal position of a point in the two images. The disparity is used to construct a 3D model of the world.

## Algorithm

The algorithm for block matching is as follows:

```python
def block_matching_imp1(imgL, imgR, window_size, method, numDisparities = 32):
    # imgL and imgR are grayscale images
    assert imgL.shape == imgR.shape
    assert imgL.ndim == 2
    assert imgR.ndim == 2
    # window_size is the size of the window
    assert window_size % 2 == 1
    # numDisparities must be less then the width of the image
    assert numDisparities < imgL.shape[1]
    # method is the method used to compute the distance between two pixels
    # method can be 'SAD' or 'SSD'
    assert method in ['SAD', 'SSD']

    win_size_over_2 = window_size//2
    imgLP = np.pad(imgL, win_size_over_2, mode='constant')
    imgRP = np.pad(imgR, win_size_over_2, mode='constant')
    disparity = np.zeros(imgL.shape, np.uint8)
    
    for r in tqdm(range(win_size_over_2, imgLP.shape[0] - win_size_over_2)):
        for p in range(win_size_over_2, imgLP.shape[1] - win_size_over_2):
            winL = imgLP[r - win_size_over_2 : r + win_size_over_2 + 1 , p - win_size_over_2 : p + win_size_over_2 + 1].astype('int32')
            min_index = win_size_over_2
            min_cost = np.Inf
            
            for i in range(p - numDisparities if p - numDisparities > win_size_over_2 else win_size_over_2, p + 1):
                winR = imgRP[r - win_size_over_2 : r + win_size_over_2 + 1 , i - win_size_over_2 : i + win_size_over_2 + 1].astype('int32')
                cost = 0
                if method == 'SAD':
                    cost = int(np.sum(np.absolute(winL - winR)))
                elif method == 'SSD':
                    cost = int(np.sum(np.square(winL - winR)))
                
                if cost <= min_cost:
                    min_cost = cost
                    min_index = i
            
            disparity[r - win_size_over_2, p - win_size_over_2] = np.absolute(int(p) - int(min_index))

    return disparity
```

## Parameters

The parameters for block matching are as follows:

1. window size

    Window size parameter tuning is usually a trade-off between smoothness and detail. A larger window size will give smoother results, but will lose detail. A smaller window size will give more detail, but will be more noisy.

    The following results show the effect of the window size.

    ![SAD window size 1][img2-SAD-1-50]

    SAD, window size = 1

    ![SSD window size 1][img2-SSD-1-50]

    SSD, window size = 1

    ![SAD window size 5][img2-SAD-5-50]

    SAD, window size = 5

    ![SSD window size 5][img2-SSD-5-50]

    SSD, window size = 5

    ![SAD window size 9][img2-SAD-9-50]

    SAD, window size = 9

    ![SSD window size 9][img2-SSD-9-50]

    SSD, window size = 9

2. search range (`numDisparities`)

    For each pixel, the search range is the number of pixels to the left of the current pixel to search for the best match. A larger search range will give more accurate results, but will be computationally expensive. A smaller search range will give less accurate results, but will be less computationally expensive.

    We search the width of the image in this setup.

    The following results show the effect of the search range.

    ![range  = 10][img1-SAD-5-50]

    SAD, range = 50

    ![range  = 10][img1-SSD-5-50]

    SSD, range = 50

    ![range  = 100][img1-SAD-5-100]

    SAD, range = 100

    ![range  = 100][img1-SSD-5-100]

    SSD, range = 100

    ![range  = 150][img1-SAD-5-150]

    SAD, range = 150

    ![range  = 150][img1-SSD-5-150]

    SSD, range = 150

### Window Size

The window size is the size of the window used to compare the blocks. The window size is a square, so the window size is the length of one side of the square.

### Search Range

The search range is the number of disparities to search for. The search range is the number of pixels to the left of the current pixel to search for the best match.

### Method

The method is the method used to compute the distance between two pixels. The method can be either `SAD` or `SSD`.

SSD is the sum of the squared differences between the two pixels. SSD is sensitive to intensity offsets.
SAD is the sum of the absolute differences between the two pixels.

Results in the previous and next sections show the effect of the method.

## Results

We show the dispaity maps for 3 images with different window sizes and methods.

### First Image

![first image left][img1-l]
![first image right][img1-r]

![img1 SAD window size 1 disparity num 50][img1-SAD-1-50]
![img1 SSD window size 1 disparity num 50][img1-SSD-1-50]
![img1 SAD window size 1 disparity num 100][img1-SAD-1-100]
![img1 SSD window size 1 disparity num 100][img1-SSD-1-100]
![img1 SAD window size 1 disparity num 150][img1-SAD-1-150]
![img1 SSD window size 1 disparity num 150][img1-SSD-1-150]

![img1 SAD window size 5 disparity num 50][img1-SAD-5-50]
![img1 SSD window size 5 disparity num 50][img1-SSD-5-50]
![img1 SAD window size 5 disparity num 100][img1-SAD-5-100]
![img1 SSD window size 5 disparity num 100][img1-SSD-5-100]
![img1 SAD window size 5 disparity num 150][img1-SAD-5-150]
![img1 SSD window size 5 disparity num 150][img1-SSD-5-150]

![img1 SAD window size 9 disparity num 50][img1-SAD-9-50]
![img1 SSD window size 9 disparity num 50][img1-SSD-9-50]
![img1 SAD window size 9 disparity num 100][img1-SAD-9-100]
![img1 SSD window size 9 disparity num 100][img1-SSD-9-100]
![img1 SAD window size 9 disparity num 150][img1-SAD-9-150]
![img1 SSD window size 9 disparity num 150][img1-SSD-9-150]

### Second Image

![second image left][img2-l]
![second image right][img2-r]

![img2 SAD window size 1 disparity num 50][img2-SAD-1-50]
![img2 SSD window size 1 disparity num 50][img2-SSD-1-50]
![img2 SAD window size 1 disparity num 100][img2-SAD-1-100]
![img2 SSD window size 1 disparity num 100][img2-SSD-1-100]
![img2 SAD window size 1 disparity num 150][img2-SAD-1-150]
![img2 SSD window size 1 disparity num 150][img2-SSD-1-150]

![img2 SAD window size 5 disparity num 50][img2-SAD-5-50]
![img2 SSD window size 5 disparity num 50][img2-SSD-5-50]
![img2 SAD window size 5 disparity num 100][img2-SAD-5-100]
![img2 SSD window size 5 disparity num 100][img2-SSD-5-100]
![img2 SAD window size 5 disparity num 150][img2-SAD-5-150]
![img2 SSD window size 5 disparity num 150][img2-SSD-5-150]

![img2 SAD window size 9 disparity num 50][img2-SAD-9-50]
![img2 SSD window size 9 disparity num 50][img2-SSD-9-50]
![img2 SAD window size 9 disparity num 100][img2-SAD-9-100]
![img2 SSD window size 9 disparity num 100][img2-SSD-9-100]
![img2 SAD window size 9 disparity num 150][img2-SAD-9-150]
![img2 SSD window size 9 disparity num 150][img2-SSD-9-150]

### Third Image

![third image left][img3-l]
![third image right][img3-r]

![img3 SAD window size 1 disparity num 50][img3-SAD-1-50]
![img3 SSD window size 1 disparity num 50][img3-SSD-1-50]
![img3 SAD window size 1 disparity num 100][img3-SAD-1-100]
![img3 SSD window size 1 disparity num 100][img3-SSD-1-100]
![img3 SAD window size 1 disparity num 150][img3-SAD-1-150]
![img3 SSD window size 1 disparity num 150][img3-SSD-1-150]

![img3 SAD window size 5 disparity num 50][img3-SAD-5-50]
![img3 SSD window size 5 disparity num 50][img3-SSD-5-50]
![img3 SAD window size 5 disparity num 100][img3-SAD-5-100]
![img3 SSD window size 5 disparity num 100][img3-SSD-5-100]
![img3 SAD window size 5 disparity num 150][img3-SAD-5-150]
![img3 SSD window size 5 disparity num 150][img3-SSD-5-150]

![img3 SAD window size 9 disparity num 50][img3-SAD-9-50]
![img3 SSD window size 9 disparity num 50][img3-SSD-9-50]
![img3 SAD window size 9 disparity num 100][img3-SAD-9-100]
![img3 SSD window size 9 disparity num 100][img3-SSD-9-100]
![img3 SAD window size 9 disparity num 150][img3-SAD-9-150]
![img3 SSD window size 9 disparity num 150][img3-SSD-9-150]

<!-- Referrences -->

[disp-10]: ./img/sad-1-10.png
[disp-100]: ./img/sad-1-100.png
[disp-10-ssd]: ./img/ssd-1-10.png
[disp-100-ssd]: ./img/ssd-1-100.png
[disp-full]: ./img/sad-1-full.png
[disp-full-ssd]: ./img/ssd-1-full.png

[win-5]: ./img/sad-full-5.png
[win-5-ssd]: ./img/ssd-full-5.png
[win-9]: ./img/sad-full-9.png
[win-9-ssd]: ./img/ssd-full-9.png

[img1-l]: ../imgs/l1.png
[img1-r]: ../imgs/r1.png
[img2-l]: ../imgs/l2.png
[img2-r]: ../imgs/r2.png
[img3-l]: ../imgs/l3.png
[img3-r]: ../imgs/r3.png

[img1-SAD-1-50]:./img/img1-SAD-1-50.png
[img1-SSD-1-50]:./img/img1-SSD-1-50.png
[img1-SAD-5-50]:./img/img1-SAD-5-50.png
[img1-SSD-5-50]:./img/img1-SSD-5-50.png
[img1-SAD-9-50]:./img/img1-SAD-9-50.png
[img1-SSD-9-50]:./img/img1-SSD-9-50.png
[img1-SAD-1-100]:./img/img1-SAD-1-100.png
[img1-SSD-1-100]:./img/img1-SSD-1-100.png
[img1-SAD-5-100]:./img/img1-SAD-5-100.png
[img1-SSD-5-100]:./img/img1-SSD-5-100.png
[img1-SAD-9-100]:./img/img1-SAD-9-100.png
[img1-SSD-9-100]:./img/img1-SSD-9-100.png
[img1-SAD-1-150]:./img/img1-SAD-1-150.png
[img1-SSD-1-150]:./img/img1-SSD-1-150.png
[img1-SAD-5-150]:./img/img1-SAD-5-150.png
[img1-SSD-5-150]:./img/img1-SSD-5-150.png
[img1-SAD-9-150]:./img/img1-SAD-9-150.png
[img1-SSD-9-150]:./img/img1-SSD-9-150.png

[img2-SAD-1-50]:./img/img2-SAD-1-50.png
[img2-SSD-1-50]:./img/img2-SSD-1-50.png
[img2-SAD-5-50]:./img/img2-SAD-5-50.png
[img2-SSD-5-50]:./img/img2-SSD-5-50.png
[img2-SAD-9-50]:./img/img2-SAD-9-50.png
[img2-SSD-9-50]:./img/img2-SSD-9-50.png
[img2-SAD-1-100]:./img/img2-SAD-1-100.png
[img2-SSD-1-100]:./img/img2-SSD-1-100.png
[img2-SAD-5-100]:./img/img2-SAD-5-100.png
[img2-SSD-5-100]:./img/img2-SSD-5-100.png
[img2-SAD-9-100]:./img/img2-SAD-9-100.png
[img2-SSD-9-100]:./img/img2-SSD-9-100.png
[img2-SAD-1-150]:./img/img2-SAD-1-150.png
[img2-SSD-1-150]:./img/img2-SSD-1-150.png
[img2-SAD-5-150]:./img/img2-SAD-5-150.png
[img2-SSD-5-150]:./img/img2-SSD-5-150.png
[img2-SAD-9-150]:./img/img2-SAD-9-150.png
[img2-SSD-9-150]:./img/img2-SSD-9-150.png

[img3-SAD-1-50]:./img/img3-SAD-1-50.png
[img3-SSD-1-50]:./img/img3-SSD-1-50.png
[img3-SAD-5-50]:./img/img3-SAD-5-50.png
[img3-SSD-5-50]:./img/img3-SSD-5-50.png
[img3-SAD-9-50]:./img/img3-SAD-9-50.png
[img3-SSD-9-50]:./img/img3-SSD-9-50.png
[img3-SAD-1-100]:./img/img3-SAD-1-100.png
[img3-SSD-1-100]:./img/img3-SSD-1-100.png
[img3-SAD-5-100]:./img/img3-SAD-5-100.png
[img3-SSD-5-100]:./img/img3-SSD-5-100.png
[img3-SAD-9-100]:./img/img3-SAD-9-100.png
[img3-SSD-9-100]:./img/img3-SSD-9-100.png
[img3-SAD-1-150]:./img/img3-SAD-1-150.png
[img3-SSD-1-150]:./img/img3-SSD-1-150.png
[img3-SAD-5-150]:./img/img3-SAD-5-150.png
[img3-SSD-5-150]:./img/img3-SSD-5-150.png
[img3-SAD-9-150]:./img/img3-SAD-9-150.png
[img3-SSD-9-150]:./img/img3-SSD-9-150.png
