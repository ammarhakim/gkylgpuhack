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
converted it to an markdown file and it is checked in here.