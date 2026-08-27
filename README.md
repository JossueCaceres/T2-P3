# Tensor++

Tarea de Programacion III hecha en C++.

La clase `Tensor` permite crear tensores de 1, 2 y 3 dimensiones. Tambien tiene operaciones como suma, resta, multiplicacion, `view`, `unsqueeze`, `concat`, `dot` y `matmul`.

El programa de `main.cpp` prueba la red neuronal que se pide en la tarea.

En CLion se debe ejecutar el objetivo `TensorPlusPlus`. El objetivo `TensorTests` sirve para correr las pruebas por separado.

## Compilar

```bash
g++ -std=c++17 Tensor.cpp main.cpp -o programa
./programa
```

## Probar

```bash
g++ -std=c++17 Tensor.cpp tests.cpp -o pruebas
./pruebas
```

Archivos:

- `Tensor.h`: declaracion de la clase.
- `Tensor.cpp`: funciones de la clase.
- `main.cpp`: ejemplo de la red neuronal.
- `tests.cpp`: algunas pruebas.
