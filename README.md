# What does this repo contain

Two illustrations of naive algorithms for matrix multiplication, one sequential as the CPU would do it, one in parallel as the GPU would do it.
The code currently uses 2x2, but it should be general enough to be modified relatively easily for larger matrices.
This is my first experiment with Manim so the code is ugly, but maybe it can be of use for you.

# How can you run it

* [Install manim](https://docs.manim.community/en/stable/installation.html) and [manim-presentation](https://www.manim.community/plugin/manim-presentation), preferably in a virtual environment.
* Clone this repository or put the `scene.py` file somewhere you have access to.
* Build the slides with

  ```bash
  $ manim -qh scene.py
  ```
  You should select both slides if you want to build both of them.

* Run the slides with

  ```bash
  $ manim-presentation MatrixMultiplication MatrixGPU
  ```

# What are you allowed to do with this repo
This repo is licensed under the MIT license, meaning you can do whatever you want with it, however I am not liable for anything if you do use it.
If you do end up using this code, attribution in your project or giving this repo a star would be appreciated.
