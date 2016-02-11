####Idea: Generating an identity function via artificial neural networks
The identity function is generated quite accurately. Sure, it makes mistakes, but converges quite fast (1st generation even) to a very good solution. Even if I feed it only 7 numbers, it usually gives a function that can calculate any number.


####Run it
This evolves the model for 100 generations

``` bash
python ident.py 100
````

####Code details
You'll see in the `ident.py` that at the end, a winner is being printed. To check out what the winner is capable of, put a breakpoint or something just before returning from the main function, and do this

``` python
winner_net.serial_activate([123])
# you should get as output a number that is close to 123... for some definition of 'close'
```
