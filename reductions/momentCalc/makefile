#.......................................................#
#.
#.Makefile to compile moment-calculation program.
#.Manaure Francisquez.
#.November 2019.
#.
#.......................................................#

COMPILER = nvcc

PROGNAME = distFuncMomentCalc1x1v
PROGEXT  = cu

CUDA_INC  = 
CUDA_LIB  = 

INCS      = $(CUDA_INC)
LIBS      = $(CUDA_LIB)

FLAGS = 

clean: $(PROGNAME).x
	rm -f $(addsuffix .mod, $(mods))

$(PROGNAME).x: $(PROGNAME).$(PROGEXT)
	$(COMPILER) $(INCS) $(LIBS) $(FLAGS) $(PROGNAME).$(PROGEXT) -o $(PROGNAME).x 

$(PROGNAME):
	echo 'Error: You need the .x extension'
