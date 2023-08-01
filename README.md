# Notes

## Transforming and augmenting images

- Transformations can be chained together using `Compose`,

- most transform classes have a function equivalent: *functional transforms* that gives fine-grained control over the transformations.
This is useful if we have to build a more complex transformation pipeline,

- the input of the transformation can be both *PIL* images and tensor images in most cases,

- the [Conversion](https://pytorch.org/vision/stable/transforms.html#conversion-transforms) can be used to convert
from and to PIL images, or for converting dtypes and ranges,

- the transformations that accept tensor images also accept batches of tensor images. In Pytorch, a Tensor image is represented as
`(C, H, W)`, and a batch of tensor images is represented as `(B, C, H, W)`,

- the same seed for torch random generator and Python random generator will not produce the same results.

### Transforms scriptability

- Use `torch.nn.Sequential` instead of `Compose` to nail the transforms scriptability
