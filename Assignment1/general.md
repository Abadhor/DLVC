
# General Information on Assignments #

The following applies to all DLVC assignments unless stated otherwise in the corresponding specifications. Make sure you understand and adhere to all the requirements, as failing to do so will cost you points.

## Groups ##

You can do the assignments alone or in groups of two. If you want to form a group, send a mail with the Matriklenummer and names of both members to [dlvc@caa.tuwien.ac.at](mailto:dlvc@caa.tuwien.ac.at) until October 20, 11pm. No group changes are allowed afterwards, and it is not possible to change groups later on.

## Programming Language ##

You can implement the assignments in Matlab, Java, Python (2 or 3), or Lua, but we suggest you use Python. We will reuse code in later assignments, so choose a language you want to use for the rest of the semester. Some assignments might require to use another language for a certain subtask.

The following image / Deep Learning libraries must be used, and no other libraries are allowed.

* Matlab: [MatConvNet](http://www.vlfeat.org/matconvnet/)
* Java: [nd4j](http://nd4j.org/) / [dl4j](http://deeplearning4j.org/index.html)
* Python: [numpy](http://www.numpy.org/) / [Keras](https://keras.io/) or [TensorFlow](https://www.tensorflow.org/)
* Lua: [torch7](http://torch.ch/)

Your code must be cross-platform, so make sure that you use only parts of your language's standard library that are available on all platforms.

## Code Templates ##

All assignments come with code templates that define the overall structure. They are in Python but easy to translate to other languages. As their main purpose is to be easy to read, they might not run without modifications.

Regardless of the language you choose, you must implement all code templates exactly as provided. For instance, if there is a template

```python
class Foo:
    # some description
    def bar(arg1, arg2):
      pass
```

then your code must have a class `Foo` that exactly matches this type in terms of interface (member names, argument names and types, and return types) and behavior (as stated in the code comments). This implies that you must use object-oriented programming, even if you use Matlab.

## Filenames ##

If there are filenames specified, you must use them as well (replace the endings depending on your language, e.g. `.m` instead of `.py`).

## Code Style ##

Your code must be easily readable and commented (similar to the provided templates).

## Indices ##

Python and Java start to count at 0, while Matlab and Lua start at 1. The assignments assume that the former is the case, e.g. the 500th image in an (ordered) dataset has ID 499. When using Matlab or Lua, you have to add 1 to all IDs given in the assignments.

## Plagiatism ##

Do not copy code from other students or the web. If we find out that you do, you will get 0 points.
