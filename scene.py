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
            element_animations = []

            for j, (re, ce) in enumerate(zip(row, col)):
                _re = re.copy()
                _ce = ce.copy()

                if last_element is None:
                    _re.next_to(mA, DOWN)
                else:
                    _re.next_to(last_element)
                re_copy = re.copy()
                g.append(re_copy)
                element_animations.append(
                    re_copy.animate.move_to(_re.get_center())
                )

                times = MathTex('\\times', color=WHITE, font_size = 20)
                times.next_to(_re)
                element_animations.append(Write(times))
                g.append(times)

                _ce.next_to(times, RIGHT)
                ce_copy = ce.copy()
                g.append(ce_copy)
                element_animations.append(
                    ce_copy.animate.move_to(_ce.get_center())
                )
                if j < n - 1:
                    plus = Tex("+", color=WHITE, font_size = 20)
                    plus.next_to(_ce, RIGHT)
                    element_animations.append(Write(plus))
                    g.append(plus)
                    last_element = plus

            gg = VGroup(*g)
            self.play(*element_animations)
            self.pause()
            self.play(Transform(gg, c_element))
            self.pause()
            self.remove(sr_r, sr_c)

        self.wait()


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
        grid.scale(0.9)
        rows_a = mA.get_rows()
        cols_b = mB.get_columns()
        rows_c = grid.get_rows()
        cols_c = grid.get_columns()
        move_ra = (
            _ra.animate.scale(0.5).next_to(_rc, LEFT).shift(1*LEFT)
            for _ra, _rc in zip(rows_a, rows_c)
        )
        move_cb = (
            _cb.animate.scale(0.5).next_to(_cc, UP).shift(0.5*UP)
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
        element_groups = []
        grid_entries = grid.get_entries()
        group_positions = []

        for i, grid_entry in enumerate(grid_entries):
            location = grid_entry.get_center()
            group_positions.append(location)

            r = i // n
            c = i % n
            row_a = rows_a[r]
            col_b = cols_b[c]

            _g_positioning = []
            _g_animation = []
            last_element = None

            for j, (re, ce) in enumerate(zip(row_a, col_b)):
                _re = re.copy()
                if last_element is not None:
                    _re.next_to(last_element)
                times = MathTex('\\times', color=WHITE, font_size = 10)
                times.next_to(_re)
                _ce = ce.copy()
                _ce.next_to(times, RIGHT)
                _g_positioning = _g_positioning + [_re, times, _ce]

                re_copy = re.copy()
                ce_copy = ce.copy()
                _g_animation = _g_animation + [re_copy, times, ce_copy]
                if j < n - 1:
                    plus = Tex("+", color=WHITE, font_size = 10)
                    plus.next_to(_ce, RIGHT)
                    _g_positioning.append(plus)
                    _g_animation.append(plus)
                    last_element = plus

            _ggpos = VGroup(*_g_positioning)
            _ggpos.move_to(location)
            for i, (pos_elem, anim_elem) in enumerate(zip(_ggpos, _g_animation)):
                if i % 2 == 0:
                    # it is an element
                    location = pos_elem.get_center()
                    motions.append(anim_elem.animate.move_to(location))
                else:
                    motions.append(Write(anim_elem))

            element_groups.append(VGroup(*_g_animation))

        self.play(FadeOut(grid_entries))
        self.play(*motions)
        self.pause()

        answers = Matrix(C).get_entries()
        answer_animations = []
        for _group, answer, pos in zip(element_groups, answers, group_positions):
            answer.move_to(pos)
            answer_animations.append(Transform(_group, answer))

        self.play(*answer_animations)
        self.pause()

        self.wait()
