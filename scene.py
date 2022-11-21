from manim import *
from manim_presentation import Slide

np.random.seed(42)
n = 2
A = np.random.randint(0, 10, size=(n, n))
B = np.random.randint(0, 10, size=(n, n))
C = A.dot(B)


class MatrixMultiplication(Slide):
    def construct(self):
        eqs = 0.8

        mA, mB, mC = Matrix(A), Matrix(B), Matrix(C)
        mA.set_color(BLUE)
        mB.set_color(RED)
        Dot = Tex(".", color=WHITE, font_size = 200)
        Equals = Tex("=", color=WHITE, font_size = 100)

        mA.scale(eqs) #.to_corner(UP + LEFT*2)
        Dot.next_to(mA, RIGHT)
        mB.scale(eqs).next_to(Dot, RIGHT)
        Equals.next_to(mB, RIGHT)
        mC.scale(eqs).next_to(Equals, RIGHT)
        equation = VGroup(mA, Dot, mB, Equals, mC)
        equation.center().to_edge(UP)

        C_elements = mC.get_entries()
        brackets = mC.get_brackets()


        # write the equation itself without an answer
        self.play(Write(mA))
        self.play(Write(Dot))
        self.play(Write(mB))
        self.play(Write(Equals))
        self.play(Write(brackets))
        self.pause()

        # definition of the serial code
        code = """def multiply(A, B):
    n_rows = A.shape[0]
    n_terms = A.shape[1]  # =B.shape[0]
    n_cols = B.shape[1]
    C = np.empty((n_rows, n_cols), dtype=np.int64)

    for row in range(n_rows):
        for col in range(n_cols):
            sum = 0

            for term in range(n_terms):
                sum += A[row, term] * B[col, term]

            C[row, col] = sum

    return C
"""
        rendered_code = Code(
            code=code,
            tab_width=4,
            language="Python",
            font="Monospace",
            insert_line_no=False,
        )
        rendered_code.scale(0.8).to_edge(DOWN)
        self.play(Write(rendered_code))
        self.pause()


        A_rows = mA.get_rows()
        B_cols = mB.get_columns()

        for i, c_element in enumerate(C_elements):
            r = i // n
            c = i % n
            row = A_rows[r]
            sr_r = SurroundingRectangle(row, color=BLUE)
            col = B_cols[c]
            sr_c = SurroundingRectangle(col, color=RED)
            last_element = None
            self.add(sr_r, sr_c)
            g = []
            for j, (re, ce) in enumerate(zip(row, col)):
                _re = re.copy()
                _ce = ce.copy()
                if last_element is None:
                    self.play(_re.animate.next_to(mA, DOWN))
                else:
                    self.play(_re.animate.next_to(last_element))
                times = MathTex('\\times', color=WHITE, font_size = 20)
                times.next_to(_re)
                self.play(Write(times))
                self.play(_ce.animate.next_to(times, RIGHT))
                g.append(_re)
                g.append(times)
                g.append(_ce)
                if j < n - 1:
                    plus = Tex("+", color=WHITE, font_size = 20)
                    plus.next_to(_ce, RIGHT)
                    self.play(Write(plus))
                    last_element = plus
                    g.append(plus)
            gg = VGroup(*g)
            self.play(Transform(gg, c_element))
            self.pause()
            self.remove(sr_r, sr_c)

        self.pause()


class MatrixGPU(Slide):
    def construct(self):
        # show the problem again but in a different way
        mA, mB, mC = Matrix(A), Matrix(B), Matrix(C)
        mA.set_color(BLUE)
        mB.set_color(RED)

        mC.center()
        mA.next_to(mC, LEFT)
        mB.next_to(mC, UP)
        mg = VGroup(mA, mB, mC)
        self.play(Write(mg))
        self.pause()

        # convert the problem to a grid
        table_values = [
            [f"({row}, {col})"for col in range(n)]
            for row in range(n)
        ]
        grid = Table(
            table_values,
            include_outer_lines=True,
        )
        grid.scale(0.8)
        rows_a = mA.get_rows()
        cols_b = mB.get_columns()
        rows_c = grid.get_rows()
        cols_c = grid.get_columns()
        move_ra = (
            _ra.animate.next_to(_rc, LEFT).shift(0.5*LEFT)
            for _ra, _rc in zip(rows_a, rows_c)
        )
        move_cb = (
            _cb.animate.next_to(_cc, UP).shift(0.5*UP)
            for _cb, _cc in zip(cols_b, cols_c)
        )

        del rows_c
        del cols_c

        self.play(
            Transform(mC, grid),
            *move_ra,
            *move_cb,
            FadeOut(mA.get_brackets()),
            FadeOut(mB.get_brackets()),
        )
        self.remove(mC)
        self.pause()
        gscene = VGroup(rows_a, cols_b, grid)

        code = """@cuda.jit
def multiply_kernel(A, B, C):
    row, col = cuda.grid(2)
    if row >= C.shape[0] or col >= C.shape[1]:
        return

    n_terms = A.shape[1]  # =B.shape[0]

    sum = 0

    for term in range(n_terms):
        sum += A[row, term] * B[col, term]

    C[row, col] = sum
"""
        rendered_code = Code(
            code=code,
            tab_width=4,
            language="Python",
            font="Monospace",
            insert_line_no=False,
        )
        rendered_code.scale(0.6).to_corner(DL)

        code2 = """C = cp.empty_like(A)
threads_per_block = (2, 2)
blocks_per_grid = (1, 1)

multiply_kernel[
    threads_per_block,
    blocks_per_grid
](A, B, C)
"""
        rendered_code_2 = Code(
            code=code2,
            tab_width=4,
            language="Python",
            font="Monospace",
            insert_line_no=False,
        )
        rendered_code_2.scale(0.8).next_to(rendered_code, RIGHT).to_edge(RIGHT)

        self.play(
            Write(rendered_code),
            gscene.animate.to_edge(UP),
            Write(rendered_code_2),
        )
        self.pause()

        motions = []
        grd_entries = grid.get_entries()

        for row_a in rows_a:
            for col_b in cols_b:
                _g = VGroup()
                last_object = None
                for i, (rea, ceb) in enumerate(zip(row_a, col_b)):
                    times = MathTex('\\times', color=WHITE, font_size = 20)
                    _g.add(rea.copy())
                    _g.add(times)
                    _g.add(ceb.copy())
                    if last_object is not None:
                        rea.next_to(last_object)
                    times.next_to(rea)
                    ceb.next_to(times)
                    if i < n - 1:
                        plus = Tex("+", color=WHITE, font_size = 20)
                        _g.add(plus)
                        plus.next_to_ceb
                        last_object = plus
                    _g.align_to(grd_entries[i])

        self.pause()
