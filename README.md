# Tensor++

**Alumno:** Caceres Molina Jossue Guillermo  
**Codigo:** 202410768

## Descripcion

Este proyecto contiene una clase `Tensor` hecha en C++. La clase trabaja con tensores de una, dos y tres dimensiones usando memoria dinamica.

Se implementaron las siguientes funciones:

- Creacion de tensores con `zeros`, `ones`, `random` y `arange`.
- Suma, resta y multiplicacion de tensores.
- Multiplicacion por un numero.
- Cambio de dimensiones con `view` y `unsqueeze`.
- Union de tensores con `concat`.
- Producto punto con `dot`.
- Multiplicacion de matrices con `matmul`.
- Funciones de activacion ReLU y Sigmoid.
- Constructor y operador de copia y movimiento.

## Red neuronal

El archivo `main.cpp` contiene el ejemplo solicitado en la tarea. Primero se crea un tensor de tamaño `1000 x 20 x 20`. Luego se cambia su forma y se realizan las multiplicaciones, sumas y funciones de activacion de la red neuronal.

El programa muestra la forma y la cantidad de elementos del tensor en cada paso.


## Archivos del proyecto

- `Tensor.h`: contiene la declaracion de la clase `Tensor`.
- `Tensor.cpp`: contiene la implementacion de sus funciones.
- `main.cpp`: contiene la red neuronal solicitada.
- `tests.cpp`: contiene pruebas de las operaciones principales.
- `CMakeLists.txt`: contiene la configuracion usada por CLion.
