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
    disparity = np.zeros(imgL.shape)

    for r in range(win_size_over_2, imgLP.shape[0] - win_size_over_2):
        for p in range(win_size_over_2, imgLP.shape[1] - win_size_over_2):
            winL = imgLP[r - win_size_over_2 : r + win_size_over_2 + 1 , p - win_size_over_2 : p + win_size_over_2 + 1].astype('int32')
            min_index = win_size_over_2
            min_cost = np.Inf

            for i in range(win_size_over_2, p + 1 - numDisparities if p > numDisparities else 0):
                winR = imgRP[r - win_size_over_2 : r + win_size_over_2 + 1 , i - win_size_over_2 : i + win_size_over_2 + 1].astype('int32')

                cost = 0
                if method == 'SAD':
                    cost = int(np.sum(np.absolute(winL - winR)))
                elif method == 'SSD':
                    cost = int(np.sum(np.square(winL - winR)))

                if cost < min_cost:
                    min_cost = cost
                    min_index = i

            disparity[r - win_size_over_2, p - win_size_over_2] = np.absolute(p - min_index)

    return disparity
```

## Parameters

The parameters for block matching are as follows:

1. window size

    Window size parameter tuning is usually a trade-off between smoothness and detail. A larger window size will give smoother results, but will lose detail. A smaller window size will give more detail, but will be more noisy.

    The following results show the effect of the window size.

    ![window size  = 1][disp-full]

    SAD, window size = 1

    ![window size  = 1][disp-full-ssd]

    SSD, window size = 1

    ![window size  = 5][win-5]

    SAD, window size = 5

    ![window size  = 5][win-5-ssd]

    SSD, window size = 5

    ![window size  = 9][win-9]

    SAD, window size = 9

    ![window size  = 9][win-9-ssd]

    SSD, window size = 9

2. search range (`numDisparities`)

    For each pixel, the search range is the number of pixels to the left and right of the current pixel to search for the best match. A larger search range will give more accurate results, but will be computationally expensive. A smaller search range will give less accurate results, but will be less computationally expensive.

    We search the width of the image in this setup.

    The following results show the effect of the search range.

    ![range  = 10][disp-10]

    SAD, range = 10

    ![range  = 10][disp-10-ssd]

    SSD, range = 10

    ![range  = 100][disp-100]

    SAD, range = 100

    ![range  = 100][disp-100-ssd]

    SSD, range = 100

    ![ Full Range][disp-full]

    SAD, full range

    ![ Full Range][disp-full-ssd]

    SSD, full range

### Window Size

The window size is the size of the window used to compare the blocks. The window size is a square, so the window size is the length of one side of the square.

### Search Range

The search range is the number of disparities to search for. The search range is the number of pixels to the left and right of the current pixel to search for the best match.

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

![sad window size 1][disp-full]

SAD, window size = 1

![ssd window size 1][disp-full-ssd]

SSD, window size = 1

![sad window size 5][win-5]

SAD, window size = 5

![ssd window size 5][win-5-ssd]

SSD, window size = 5

![sad window size 9][win-9]

SAD, window size = 9

![ssd window size 9][win-9-ssd]

SSD, window size = 9

### Second Image

![second image left][img2-l]
![second image right][img2-r]

![sad 2 window size 1][sad-2-1]
![ssd 2 window size 1][ssd-2-1]

![sad 2 window size 5][sad-2-5]
![ssd 2 window size 5][ssd-2-5]

![sad 2 window size 9][sad-2-9]
![ssd 2 window size 9][ssd-2-9]

### Third Image

![third image left][img3-l]
![third image right][img3-r]

![sad 3 window size 1][sad-3-1]
![ssd 3 window size 1][ssd-3-1]

![sad 3 window size 5][sad-3-5]
![ssd 3 window size 5][ssd-3-5]

![sad 3 window size 9][sad-3-9]
![ssd 3 window size 9][ssd-3-9]

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

[sad-2-1]: ./img/sad-2-1.png
[ssd-2-1]: ./img/ssd-2-1.png

[sad-2-5]: ./img/sad-2-5.png
[ssd-2-5]: ./img/ssd-2-5.png

[sad-2-9]: ./img/sad-2-9.png
[ssd-2-9]: ./img/ssd-2-9.png

[sad-3-1]: ./img/sad-3-1.png
[ssd-3-1]: ./img/ssd-3-1.png

[sad-3-5]: ./img/sad-3-5.png
[ssd-3-5]: ./img/ssd-3-5.png

[sad-3-9]: ./img/sad-3-9.png
[ssd-3-9]: ./img/ssd-3-9.png
