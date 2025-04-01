from skimage import io

map_array = io.imread("711_casia.bmp")
map_array = map_array.mean(axis=2)
io.imsave("711_casia.pgm", map_array)