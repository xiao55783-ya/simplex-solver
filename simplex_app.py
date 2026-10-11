import math
import flet as ft

EPS = 1e-9


# =========================================================
# SIMPLEX UTILITIES
# =========================================================

def pivot(tableau, basis, row_index, col_index):
    """Perform a Gauss-Jordan pivot."""
    value = tableau[row_index][col_index]

    if abs(value) < EPS:
        raise ArithmeticError("Numerically unstable pivot.")

    tableau[row_index] = [v / value for v in tableau[row_index]]

    for i in range(len(tableau)):
        if i == row_index:
            continue

        factor = tableau[i][col_index]

        if abs(factor) > EPS:
            tableau[i] = [
                tableau[i][j] - factor * tableau[row_index][j]
                for j in range(len(tableau[i]))
            ]

    basis[row_index] = col_index


def run_simplex(tableau, basis, costs, variable_count):
    """
    Primal simplex for a feasible starting basis.
    Uses Bland's rule to reduce cycling.
    """
    row_count = len(tableau) - 1

    for _ in range(10000):
        # Build the canonical objective row.
        objective = [-costs[j] for j in range(variable_count)] + [0.0]

        for i in range(row_count):
            cb = costs[basis[i]]

            if abs(cb) > EPS:
                objective = [
                    objective[j] + cb * tableau[i][j]
                    for j in range(variable_count + 1)
                ]

        tableau[-1] = objective

        # Bland's rule: first negative reduced cost enters.
        entering = next(
            (
                j for j in range(variable_count)
                if tableau[-1][j] < -EPS
            ),
            None
        )

        if entering is None:
            return "optimal"

        leaving = None
        best_ratio = math.inf

        for i in range(row_count):
            coefficient = tableau[i][entering]

            if coefficient > EPS:
                ratio = tableau[i][-1] / coefficient

                if ratio < best_ratio - EPS:
                    best_ratio = ratio
                    leaving = i
                elif (
                    abs(ratio - best_ratio) <= EPS
                    and leaving is not None
                    and basis[i] < basis[leaving]
                ):
                    leaving = i

        if leaving is None:
            return "unbounded"

        pivot(tableau, basis, leaving, entering)

    raise RuntimeError(
        "Iteration limit reached. The problem may be numerically difficult."
    )


# =========================================================
# TWO-PHASE SIMPLEX SOLVER
# =========================================================

def solve_simplex(c1, c2, constraints, obj_type):
    """
    Two-phase simplex for two nonnegative decision variables.

    Supports:
      Max and Min
      <=, >=, =
      Negative right-hand sides
      Infeasible and unbounded problems

    Returns:
      (x1, x2, objective_value)
      (None, None, "Infeasible")
      (None, None, "Unbounded")
    """

    c1 = float(c1)
    c2 = float(c2)

    if not math.isfinite(c1) or not math.isfinite(c2):
        raise ValueError("Objective coefficients must be finite numbers.")

    if obj_type not in ("Max", "Min"):
        raise ValueError("Choose Max or Min.")

    if not constraints:
        raise ValueError("Add at least one constraint.")

    # Normalize each constraint.
    normalized = []

    for number, constraint in enumerate(constraints, start=1):
        try:
            a = float(constraint["c1"])
            b = float(constraint["c2"])
            rhs = float(constraint["rhs"])
            sign = constraint["sign"]
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"Invalid values in constraint {number}.")

        if not all(math.isfinite(v) for v in (a, b, rhs)):
            raise ValueError(
                f"Constraint {number} must use finite numbers."
            )

    

        if sign not in ("<=", ">=", "="):
            raise ValueError(f"Invalid sign in constraint {number}.")

        # Make RHS nonnegative and reverse the inequality when needed.
        if rhs < -EPS:
            a, b, rhs = -a, -b, -rhs

            if sign == "<=":
                sign = ">="
            elif sign == ">=":
                sign = "<="

        normalized.append((a, b, sign, rhs))

    # x1 and x2 are nonnegative by definition.
    # Standard form columns: x1, x2, then slack/surplus/artificial.
    variable_count = 2
    specs = []
    artificial_columns = set()

    for a, b, sign, rhs in normalized:
        spec = {
            "a": a,
            "b": b,
            "sign": sign,
            "rhs": rhs,
            "slack": None,
            "surplus": None,
            "artificial": None,
        }

        if sign == "<=":
            spec["slack"] = variable_count
            variable_count += 1

        elif sign == ">=":
            spec["surplus"] = variable_count
            variable_count += 1

            spec["artificial"] = variable_count
            artificial_columns.add(variable_count)
            variable_count += 1

        else:  # Equality constraint
            spec["artificial"] = variable_count
            artificial_columns.add(variable_count)
            variable_count += 1

        specs.append(spec)

    tableau = []
    basis = []

    for spec in specs:
        row = [0.0] * (variable_count + 1)

        row[0] = spec["a"]
        row[1] = spec["b"]
        row[-1] = spec["rhs"]

        if spec["slack"] is not None:
            row[spec["slack"]] = 1.0
            basis.append(spec["slack"])

        elif spec["artificial"] is not None:
            if spec["surplus"] is not None:
                row[spec["surplus"]] = -1.0

            row[spec["artificial"]] = 1.0
            basis.append(spec["artificial"])

        tableau.append(row)

    # Phase I: maximize minus the sum of artificial variables.
    tableau.append([0.0] * (variable_count + 1))

    phase_one_costs = [0.0] * variable_count

    for col in artificial_columns:
        phase_one_costs[col] = -1.0

    status = run_simplex(
        tableau, basis, phase_one_costs, variable_count
    )

    if status == "unbounded":
        raise RuntimeError("Unexpected Phase I result.")

    # The Phase I optimum must be zero for a feasible problem.
    phase_one_value = tableau[-1][-1]

    if phase_one_value < -1e-7:
        return None, None, "Infeasible"

    # Remove artificial basic variables.
    i = 0

    while i < len(basis):
        if basis[i] not in artificial_columns:
            i += 1
            continue

        entering = next(
            (
                j for j in range(variable_count)
                if j not in artificial_columns
                and abs(tableau[i][j]) > EPS
            ),
            None
        )

        if entering is not None:
            pivot(tableau, basis, i, entering)
            i += 1
        else:
            # A zero row with an artificial basic variable is redundant.
            if abs(tableau[i][-1]) > 1e-7:
                return None, None, "Infeasible"

            del tableau[i]
            del basis[i]

    # Remove artificial columns and remap indices.
    keep = [
        j for j in range(variable_count)
        if j not in artificial_columns
    ]

    old_to_new = {old: new for new, old in enumerate(keep)}

    tableau = [
        [row[j] for j in keep] + [row[-1]]
        for row in tableau
    ]

    basis = [old_to_new[col] for col in basis]
    variable_count = len(keep)

    # Phase II: optimize the real objective.
    direction = 1.0 if obj_type == "Max" else -1.0
    costs = [0.0] * variable_count

    costs[old_to_new[0]] = direction * c1
    costs[old_to_new[1]] = direction * c2

    status = run_simplex(tableau, basis, costs, variable_count)

    if status == "unbounded":
        return None, None, "Unbounded"

    values = [0.0] * variable_count

    for i, basic_col in enumerate(basis):
        values[basic_col] = tableau[i][-1]

    x1 = values[old_to_new[0]]
    x2 = values[old_to_new[1]]
    z = c1 * x1 + c2 * x2

    # Remove tiny floating-point noise.
    if abs(x1) < 1e-7:
        x1 = 0.0
    if abs(x2) < 1e-7:
        x2 = 0.0
    if abs(z) < 1e-7:
        z = 0.0

    return x1, x2, z


# =========================================================
# FLET USER INTERFACE
# =========================================================

def main(page: ft.Page):
    BG = "#101820"
    PANEL = "#1B2A36"
    BORDER = "#354A5A"
    TEXT = "#F4F7FB"
    MUTED = "#B7C5D1"
    BLUE = "#2689E8"
    GREEN_BG = "#173C32"
    GREEN_BORDER = "#3FA77A"
    GREEN_TEXT = "#82E6B0"
    RED = "#FF7777"
    FIELD_BG = "#14232D"

    page.title = "Simplexe Solver"
    page.bgcolor = BG
    page.padding = 18
    page.vertical_alignment = ft.MainAxisAlignment.START
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.scroll = ft.ScrollMode.AUTO

    title_text = ft.Text(
        "Objective Function",
        size=22,
        weight=ft.FontWeight.BOLD,
        color=TEXT,
    )

    def make_field(label, width):
        return ft.TextField(
            label=label,
            width=width,
            text_size=15,
            color=TEXT,
            label_style=ft.TextStyle(color=MUTED),
            border_color=BORDER,
            focused_border_color=BLUE,
            bgcolor=FIELD_BG,
            border_radius=10,
        )

    x1_input = make_field("x1 coefficient", 150)
    x2_input = make_field("x2 coefficient", 150)

    inputs_row = ft.Row(
        [x1_input, x2_input],
        alignment=ft.MainAxisAlignment.CENTER,
        spacing=10,
        wrap=True,
    )

    selected_objective = "Max"

    def objective_clicked(e):
        nonlocal selected_objective
        selected_objective = e.control.text

        Max_btn.bgcolor = BLUE if selected_objective == "Max" else PANEL
        Max_btn.color = TEXT if selected_objective == "Max" else MUTED

        Min_btn.bgcolor = BLUE if selected_objective == "Min" else PANEL
        Min_btn.color = TEXT if selected_objective == "Min" else MUTED

        page.update()

    Max_btn = ft.ElevatedButton(
        "Max",
        bgcolor=BLUE,
        color=TEXT,
        width=120,
        on_click=objective_clicked,
    )

    Min_btn = ft.ElevatedButton(
        "Min",
        bgcolor=PANEL,
        color=MUTED,
        width=120,
        on_click=objective_clicked,
    )

    objective_row = ft.Row(
        [Max_btn, Min_btn],
        alignment=ft.MainAxisAlignment.CENTER,
        spacing=12,
    )

    constraint_x1 = make_field("x1", 65)
    constraint_x2 = make_field("x2", 65)
    rhs_input = make_field("RHS", 75)

    sign_dropdown = ft.Dropdown(
        options=[
            ft.dropdown.Option("<="),
            ft.dropdown.Option(">="),
            ft.dropdown.Option("="),
        ],
        value="<=",
        width=72,
        color=TEXT,
        bgcolor=FIELD_BG,
        border_color=BORDER,
        focused_border_color=BLUE,
    )

    constraints = []
    constraints_column = ft.Column(spacing=7, tight=True)

    result_display = ft.Text(
        "Z = --", size=25, weight=ft.FontWeight.BOLD, color=GREEN_TEXT
    )

    details_display = ft.Text(
        "x1 = -- , x2 = --",
        size=16,
        color=TEXT,
        text_align=ft.TextAlign.CENTER,
    )

    status_display = ft.Text(
        "", size=13, color=MUTED, text_align=ft.TextAlign.CENTER
    )

    # =====================================================
    # CONSTRAINT DISPLAY FORMATTING
    # =====================================================

    def format_number(value):
        """Display numbers without unnecessary trailing zeros."""
        number = float(value)

        if abs(number) < EPS:
            number = 0.0

        return f"{number:.10g}"

    def format_constraint(constraint):
        """Format a constraint as a clean mathematical expression."""
        terms = []

        for coefficient_key, variable in (
            ("c1", "x1"),
            ("c2", "x2"),
        ):
            coefficient = float(constraint[coefficient_key])

            # Hide variables whose coefficients are zero.
            if abs(coefficient) < EPS:
                continue

            magnitude = abs(coefficient)

            # Do not display the coefficient 1 or -1.
            if abs(magnitude - 1.0) < EPS:
                term = variable
            else:
                term = f"{format_number(magnitude)}{variable}"

            if not terms:
                # The first term has no leading plus sign.
                if coefficient < 0:
                    term = "-" + term
            else:
                # Display the correct sign between terms.
                term = (" - " if coefficient < 0 else " + ") + term

            terms.append(term)

        # If both coefficients are zero, display zero.
        left_side = "".join(terms) if terms else "0"

        sign_symbols = {
            "<=": "≤",
            ">=": "≥",
            "<": "<",
            ">": ">",
            "=": "=",
        }

        relation = sign_symbols.get(
            constraint["sign"],
            constraint["sign"],
        )

        rhs = format_number(constraint["rhs"])

        return f"{left_side} {relation} {rhs}"

    def update_constraints_ui():
        constraints_column.controls.clear()

        for index, constraint in enumerate(constraints):

            def remove_constraint(e, idx=index):
                constraints.pop(idx)
                update_constraints_ui()
                page.update()

            description = (
                f"({index + 1}) "
                f"{format_constraint(constraint)}"
            )

            constraints_column.controls.append(
                ft.Row(
                    [
                        ft.Text(
                            description,
                            size=13,
                            color=TEXT,
                            expand=True,
                        ),
                        ft.IconButton(
                            icon=ft.icons.DELETE_OUTLINE,
                            icon_color=RED,
                            tooltip="Remove constraint",
                            on_click=remove_constraint,
                        ),
                    ],
                    width=310,
                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                )
            )

    def close_dialog(e):
        constraint_dialog.open = False
        page.update()

    def save_constraint(e):
        a_text = (constraint_x1.value or "").strip()
        b_text = (constraint_x2.value or "").strip()
        rhs_text = (rhs_input.value or "").strip()

        try:
            if not a_text or not b_text or not rhs_text:
                raise ValueError(
                    "Enter both coefficients and the RHS."
                )

            numbers = [
                float(a_text),
                float(b_text),
                float(rhs_text),
            ]

            if not all(math.isfinite(v) for v in numbers):
                raise ValueError("Use finite numeric values.")

        except ValueError as ex:
            status_display.value = str(ex)
            status_display.color = RED
            page.update()
            return

        constraints.append({
            "c1": a_text,
            "c2": b_text,
            "sign": sign_dropdown.value,
            "rhs": rhs_text,
        })

        constraint_x1.value = ""
        constraint_x2.value = ""
        rhs_input.value = ""
        sign_dropdown.value = "<="

        constraint_dialog.open = False
        status_display.value = ""
        update_constraints_ui()
        page.update()

    constraint_dialog = ft.AlertDialog(
        modal=True,
        title=ft.Text(
            "Add constraint",
            color=TEXT,
            weight=ft.FontWeight.BOLD,
        ),
        bgcolor=PANEL,
        content=ft.Row(
            [
                constraint_x1,
                constraint_x2,
                sign_dropdown,
                rhs_input,
            ],
            alignment=ft.MainAxisAlignment.CENTER,
            spacing=4,
            wrap=True,
        ),
        actions=[
            ft.TextButton("Cancel", on_click=close_dialog),
            ft.ElevatedButton(
                "Add",
                bgcolor=BLUE,
                color=TEXT,
                on_click=save_constraint,
            ),
        ],
        actions_alignment=ft.MainAxisAlignment.END,
    )

    def open_dialog(e):
        if constraint_dialog not in page.overlay:
            page.overlay.append(constraint_dialog)

        constraint_dialog.open = True
        page.update()

    add_constraint_button = ft.TextButton(
        "＋ Add Constraint",
        on_click=open_dialog,
    )

    constraints_container = ft.Container(
        content=ft.Column(
            [
                ft.Text(
                    "Constraints",
                    size=20,
                    color=TEXT,
                    weight=ft.FontWeight.BOLD,
                ),
                constraints_column,
                add_constraint_button,
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=12,
        ),
        padding=16,
        border=ft.border.all(1, BORDER),
        border_radius=14,
        bgcolor=PANEL,
        width=350,
    )

    def calculate_clicked(e):
        try:
            if not (x1_input.value or "").strip():
                raise ValueError(
                    "Enter the x1 objective coefficient."
                )

            if not (x2_input.value or "").strip():
                raise ValueError(
                    "Enter the x2 objective coefficient."
                )

            if not constraints:
                raise ValueError(
                    "Please add at least one constraint."
                )

            c1 = float(x1_input.value)
            c2 = float(x2_input.value)

            x1, x2, z = solve_simplex(
                c1,
                c2,
                constraints,
                selected_objective,
            )

            if z == "Infeasible":
                result_display.value = "No feasible solution"
                details_display.value = (
                    "The constraints are inconsistent."
                )
                status_display.value = "Check the constraints."
                status_display.color = RED

            elif z == "Unbounded":
                result_display.value = "Unbounded"
                details_display.value = "No finite optimum exists."
                status_display.value = (
                    "The objective can improve indefinitely."
                )
                status_display.color = RED

            else:
                result_display.value = (
                    f"{selected_objective} Z = {z:.2f}"
                )
                details_display.value = (
                    f"x1 = {x1:.2f} , x2 = {x2:.2f}"
                )
                status_display.value = "Optimal solution found."
                status_display.color = GREEN_TEXT

        except (ValueError, RuntimeError, ArithmeticError) as ex:
            result_display.value = "Cannot solve"
            details_display.value = str(ex)
            status_display.value = "Check your input values."
            status_display.color = RED

        page.update()

    calculate_button = ft.ElevatedButton(
        "Calculate",
        bgcolor=BLUE,
        color=TEXT,
        width=230,
        height=52,
        on_click=calculate_clicked,
    )

    result_container = ft.Container(
        content=ft.Column(
            [
                ft.Text(
                    "Result",
                    size=16,
                    color=MUTED,
                    weight=ft.FontWeight.BOLD,
                ),
                result_display,
                details_display,
                status_display,
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=8,
        ),
        padding=18,
        border=ft.border.all(1, GREEN_BORDER),
        border_radius=14,
        bgcolor=GREEN_BG,
        width=350,
    )

    page.add(
        title_text,
        inputs_row,
        ft.Container(height=5),
        objective_row,
        ft.Container(height=12),
        constraints_container,
        ft.Container(height=12),
        calculate_button,
        ft.Container(height=12),
        result_container,
    )

    page.update()


ft.app(main)
