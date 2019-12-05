# What

We can potentially use the Maxima gentran package to generate the C++
code needed. This package helps convert Maxima code to C++ and, most
important for us, allows the use of templates in which the C++ code
and Maxima code can be interleaved.

Unfortunately, the gentran package in the current version of Maxima is
broken for C code generation. On posting a query on the Maxima mailing
list Michael Stern (from NIH) sent me his fix to the package. The
working Lisp code is in the file allgentran.lisp. To use this, please
add this file to your share/contrib/gentran directory. On my Mac with
Maxima 5.43.0 the exact location of this directory is:

```
/Applications/Maxima.app/Contents/Resources/opt/share/maxima/5.43.0/share/contrib/gentran
```

Michael also sent me his version of the Gentran user manual. I
converted it to an markdown file and it is checked in here. I strongly
recommend everyone updates Maxima to the latest 5.43.0 version before
the hackathon.

# How to use

Template files can live anywhere in the Maxima load path. As most of
us have cas-scripts in the path, this means we can store the templates
in the sub-directories of cas-scripts and use them. For example, say a
template ```fpo.template``` is stored in the directory ```fpo```. Then
to run the template do:

```
close(openw("~/max-out/fpo.cpp"))$
gentranin("fpo/fpo.template", ["~/max-out/fpo.cpp"])$
```

The first command will open the file and delete its contents. This is
needed as ```gentranin``` appeands to the open file everytime it is
called and does not overwrite its contents.


