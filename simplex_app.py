
import math
import flet as ft

EPS = 1e-9
MAX_VARIABLES = 5


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
    """Simplex method using Bland's rule."""
    row_count = len(tableau) - 1

    for _ in range(10000):
        # Construct the objective row.
        objective = [-costs[j] for j in range(variable_count)] + [0.0]

        for i in range(row_count):
            cb = costs[basis[i]]

            if abs(cb) > EPS:
                objective = [
                    objective[j] + cb * tableau[i][j]
                    for j in range(variable_count + 1)
                ]

        tableau[-1] = objective

        # Select the first variable with a negative reduced cost.
        entering = next(
            (
                j for j in range(variable_count)
                if tableau[-1][j] < -EPS
            ),
            None
        )

        if entering is None:
            return "optimal"

        # Minimum-ratio test.
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
# TWO-PHASE SIMPLEX SOLVER: 2 TO 5 VARIABLES
# =========================================================

def solve_simplex(objective_coeffs, constraints, obj_type):
    """
    Two-phase simplex for nonnegative decision variables.

    objective_coeffs:
        List of objective coefficients, e.g. [3, 5, 2].

    constraints:
        List of dictionaries:
        {
            "coefficients": [2, 1, 3],
            "sign": "<=",
            "rhs": 10
        }

    Supports Max, Min, <=, >=, =, negative RHS,
    infeasible problems, and unbounded problems.
    """

    n = len(objective_coeffs)

    if not 2 <= n <= MAX_VARIABLES:
        raise ValueError("Choose between 2 and 5 variables.")

    if obj_type not in ("Max", "Min"):
        raise ValueError("Choose Max or Min.")

    try:
        objective_coeffs = [float(c) for c in objective_coeffs]
    except (TypeError, ValueError):
        raise ValueError("Objective coefficients must be numbers.")

    if not all(math.isfinite(c) for c in objective_coeffs):
        raise ValueError("Objective coefficients must be finite numbers.")

    if not constraints:
        raise ValueError("Add at least one constraint.")

    # Normalize the constraints.
    normalized = []

    for number, constraint in enumerate(constraints, start=1):
        try:
            coefficients = [
                float(c) for c in constraint["coefficients"]
            ]
            rhs = float(constraint["rhs"])
            sign = constraint["sign"]
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"Invalid values in constraint {number}.")

        if len(coefficients) != n:
            raise ValueError(
                f"Constraint {number} must contain {n} coefficients."
            )

        if not all(math.isfinite(c) for c in coefficients):
            raise ValueError(
                f"Constraint {number} must use finite coefficients."
            )

        if not math.isfinite(rhs):
            raise ValueError(
                f"Constraint {number} must have a finite RHS."
            )

        if sign not in ("<=", ">=", "="):
            raise ValueError(f"Invalid sign in constraint {number}.")

        # Make the RHS nonnegative.
        if rhs < -EPS:
            coefficients = [-c for c in coefficients]
            rhs = -rhs

            if sign == "<=":
                sign = ">="
            elif sign == ">=":
                sign = "<="

        normalized.append((coefficients, sign, rhs))

    # First n columns are the decision variables.
    variable_count = n
    specs = []
    artificial_columns = set()

    # Assign slack, surplus, and artificial variables.
    for coefficients, sign, rhs in normalized:
        spec = {
            "coefficients": coefficients,
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

        else:  # Equality
            spec["artificial"] = variable_count
            artificial_columns.add(variable_count)
            variable_count += 1

        specs.append(spec)

    # Build the initial tableau.
    tableau = []
    basis = []

    for spec in specs:
        row = [0.0] * (variable_count + 1)

        for j, coefficient in enumerate(spec["coefficients"]):
            row[j] = coefficient

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

    tableau.append([0.0] * (variable_count + 1))

    # -----------------------------------------------------
    # PHASE I: Minimize the sum of artificial variables.
    # Equivalently, maximize its negative.
    # -----------------------------------------------------

    phase_one_costs = [0.0] * variable_count

    for col in artificial_columns:
        phase_one_costs[col] = -1.0

    status = run_simplex(
        tableau, basis, phase_one_costs, variable_count
    )

    if status == "unbounded":
        raise RuntimeError("Unexpected Phase I result.")

    phase_one_value = tableau[-1][-1]

    if phase_one_value < -1e-7:
        return None, "Infeasible"

    # Remove artificial variables from the basis.
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
                return None, "Infeasible"

            del tableau[i]
            del basis[i]

    # Remove artificial columns and remap the remaining indices.
    keep = [
        j for j in range(variable_count)
        if j not in artificial_columns
    ]

    old_to_new = {
        old: new for new, old in enumerate(keep)
    }

    tableau = [
        [row[j] for j in keep] + [row[-1]]
        for row in tableau
    ]

    basis = [old_to_new[col] for col in basis]
    variable_count = len(keep)

    # -----------------------------------------------------
    # PHASE II: Optimize the real objective.
    # -----------------------------------------------------

    direction = 1.0 if obj_type == "Max" else -1.0
    costs = [0.0] * variable_count

    for j, coefficient in enumerate(objective_coeffs):
        costs[old_to_new[j]] = direction * coefficient

    status = run_simplex(tableau, basis, costs, variable_count)

    if status == "unbounded":
        return None, "Unbounded"

    # Extract the values of the decision variables.
    values = [0.0] * variable_count

    for i, basic_col in enumerate(basis):
        values[basic_col] = tableau[i][-1]

    solution = []

    for j in range(n):
        value = values[old_to_new[j]]

        if abs(value) < 1e-7:
            value = 0.0

        solution.append(value)

    z = sum(
        objective_coeffs[j] * solution[j]
        for j in range(n)
    )

    if abs(z) < 1e-7:
        z = 0.0

    return (solution, z), "Optimal"


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

    def make_field(label, width=90):
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
            keyboard_type=ft.KeyboardType.NUMBER,
        )

    title_text = ft.Text(
        "Simplexe Solver",
        size=24,
        weight=ft.FontWeight.BOLD,
        color=TEXT,
    )

    # -----------------------------------------------------
    # NUMBER OF DECISION VARIABLES
    # -----------------------------------------------------

    variable_count_dropdown = ft.Dropdown(
        label="Number of variables",
        width=210,
        value="2",
        options=[
            ft.dropdown.Option(str(i)) for i in range(2, 6)
        ],
        color=TEXT,
        label_style=ft.TextStyle(color=MUTED),
        bgcolor=FIELD_BG,
        border_color=BORDER,
        focused_border_color=BLUE,
    )

    objective_title = ft.Text(
        "Objective Function",
        size=20,
        weight=ft.FontWeight.BOLD,
        color=TEXT,
    )

    objective_inputs_row = ft.Row(
        alignment=ft.MainAxisAlignment.CENTER,
        spacing=8,
        wrap=True,
    )

    objective_inputs = []

    # -----------------------------------------------------
    # MAX / MIN SELECTION
    # -----------------------------------------------------

    selected_objective = "Max"

    def objective_clicked(e):
        nonlocal selected_objective

        selected_objective = e.control.text

        max_btn.bgcolor = (
            BLUE if selected_objective == "Max" else PANEL
        )
        max_btn.color = (
            TEXT if selected_objective == "Max" else MUTED
        )

        min_btn.bgcolor = (
            BLUE if selected_objective == "Min" else PANEL
        )
        min_btn.color = (
            TEXT if selected_objective == "Min" else MUTED
        )

        page.update()

    max_btn = ft.ElevatedButton(
        "Max",
        bgcolor=BLUE,
        color=TEXT,
        width=120,
        on_click=objective_clicked,
    )

    min_btn = ft.ElevatedButton(
        "Min",
        bgcolor=PANEL,
        color=MUTED,
        width=120,
        on_click=objective_clicked,
    )

    objective_row = ft.Row(
        [max_btn, min_btn],
        alignment=ft.MainAxisAlignment.CENTER,
        spacing=12,
    )

    # -----------------------------------------------------
    # CONSTRAINT INPUTS
    # -----------------------------------------------------

    constraint_coefficient_inputs = []

    rhs_input = make_field("RHS", 85)

    sign_dropdown = ft.Dropdown(
        options=[
            ft.dropdown.Option("<="),
            ft.dropdown.Option(">="),
            ft.dropdown.Option("="),
        ],
        value="<=",
        width=85,
        color=TEXT,
        bgcolor=FIELD_BG,
        border_color=BORDER,
        focused_border_color=BLUE,
    )

    constraint_inputs_row = ft.Row(
        alignment=ft.MainAxisAlignment.CENTER,
        spacing=5,
        wrap=True,
    )

    constraints = []
    constraints_column = ft.Column(spacing=7, tight=True)

    result_display = ft.Text(
        "Z = --",
        size=25,
        weight=ft.FontWeight.BOLD,
        color=GREEN_TEXT,
        text_align=ft.TextAlign.CENTER,
    )

    details_display = ft.Text(
        "Decision variables: --",
        size=16,
        color=TEXT,
        text_align=ft.TextAlign.CENTER,
    )

    status_display = ft.Text(
        "",
        size=13,
        color=MUTED,
        text_align=ft.TextAlign.CENTER,
    )

    # -----------------------------------------------------
    # FORMATTING
    # -----------------------------------------------------

    def format_number(value):
        value = float(value)

        if abs(value) < EPS:
            value = 0.0

        return f"{value:.10g}"

    def format_constraint(constraint):
        terms = []

        for j, raw_coefficient in enumerate(
            constraint["coefficients"]
        ):
            coefficient = float(raw_coefficient)

            if abs(coefficient) < EPS:
                continue

            variable = f"x{j + 1}"
            magnitude = abs(coefficient)

            if abs(magnitude - 1.0) < EPS:
                term = variable
            else:
                term = f"{format_number(magnitude)}{variable}"

            if not terms:
                if coefficient < 0:
                    term = "-" + term
            else:
                term = (
                    " - " if coefficient < 0 else " + "
                ) + term

            terms.append(term)

        left_side = "".join(terms) if terms else "0"

        symbols = {
            "<=": "≤",
            ">=": "≥",
            "=": "=",
        }

        relation = symbols[constraint["sign"]]

        return (
            f"{left_side} {relation} "
            f"{format_number(constraint['rhs'])}"
        )

    # -----------------------------------------------------
    # DYNAMIC OBJECTIVE INPUTS
    # -----------------------------------------------------

    def rebuild_objective_inputs():
        nonlocal objective_inputs

        n = int(variable_count_dropdown.value)

        objective_inputs = [
            make_field(f"x{i + 1}", 85)
            for i in range(n)
        ]

        objective_inputs_row.controls = objective_inputs
        rebuild_constraint_inputs()
        page.update()

    # -----------------------------------------------------
    # DYNAMIC CONSTRAINT INPUTS
    # -----------------------------------------------------

    def rebuild_constraint_inputs():
        nonlocal constraint_coefficient_inputs

        n = int(variable_count_dropdown.value)

        constraint_coefficient_inputs = [
            make_field(f"x{i + 1}", 65)
            for i in range(n)
        ]

        constraint_inputs_row.controls = (
            constraint_coefficient_inputs
            + [sign_dropdown, rhs_input]
        )

        if "constraint_dialog" in globals():
            pass

        if constraint_dialog in page.overlay:
            constraint_dialog.content = constraint_inputs_row

        page.update()

    # -----------------------------------------------------
    # DISPLAY CONSTRAINTS
    # -----------------------------------------------------

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
                    width=350,
                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                )
            )

    # -----------------------------------------------------
    # DIALOG ACTIONS
    # -----------------------------------------------------

    def close_dialog(e):
        constraint_dialog.open = False
        page.update()

    def save_constraint(e):
        try:
            coefficient_texts = [
                (field.value or "").strip()
                for field in constraint_coefficient_inputs
            ]

            rhs_text = (rhs_input.value or "").strip()

            if any(not value for value in coefficient_texts):
                raise ValueError(
                    "Enter every constraint coefficient."
                )

            if not rhs_text:
                raise ValueError("Enter the RHS value.")

            coefficients = [
                float(value) for value in coefficient_texts
            ]
            rhs = float(rhs_text)

            if not all(
                math.isfinite(value)
                for value in coefficients + [rhs]
            ):
                raise ValueError("Use finite numeric values.")

        except ValueError as ex:
            status_display.value = str(ex)
            status_display.color = RED
            page.update()
            return

        constraints.append({
            "coefficients": coefficients,
            "sign": sign_dropdown.value,
            "rhs": rhs,
        })

        for field in constraint_coefficient_inputs:
            field.value = ""

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
            [],
            alignment=ft.MainAxisAlignment.CENTER,
            spacing=5,
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

        rebuild_constraint_inputs()
        constraint_dialog.open = True
        page.update()

    add_constraint_button = ft.TextButton(
        "＋ Add Constraint",
        on_click=open_dialog,
    )

    # -----------------------------------------------------
    # CONSTRAINTS PANEL
    # -----------------------------------------------------

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
        width=380,
    )

    # -----------------------------------------------------
    # CALCULATE
    # -----------------------------------------------------

    def calculate_clicked(e):
        try:
            objective_texts = [
                (field.value or "").strip()
                for field in objective_inputs
            ]

            if any(not value for value in objective_texts):
                raise ValueError(
                    "Enter every objective coefficient."
                )

            objective_coeffs = [
                float(value) for value in objective_texts
            ]

            if not all(
                math.isfinite(value)
                for value in objective_coeffs
            ):
                raise ValueError(
                    "Objective coefficients must be finite numbers."
                )

            if not constraints:
                raise ValueError(
                    "Please add at least one constraint."
                )

            result, status = solve_simplex(
                objective_coeffs,
                constraints,
                selected_objective,
            )

            if status == "Infeasible":
                result_display.value = "No feasible solution"
                details_display.value = (
                    "The constraints are inconsistent."
                )
                status_display.value = "Check the constraints."
                status_display.color = RED

            elif status == "Unbounded":
                result_display.value = "Unbounded"
                details_display.value = "No finite optimum exists."
                status_display.value = (
                    "The objective can improve indefinitely."
                )
                status_display.color = RED

            else:
                solution, z = result

                result_display.value = (
                    f"{selected_objective} Z = {z:.6g}"
                )

                details_display.value = "   |   ".join(
                    f"x{i + 1} = {value:.6g}"
                    for i, value in enumerate(solution)
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
        width=380,
    )

    # -----------------------------------------------------
    # VARIABLE COUNT CHANGE
    # -----------------------------------------------------

    variable_count_dropdown.on_change = (
        lambda e: rebuild_objective_inputs()
    )

    # -----------------------------------------------------
    # BUILD PAGE
    # -----------------------------------------------------

    page.add(
        title_text,
        variable_count_dropdown,
        objective_title,
        objective_inputs_row,
        ft.Container(height=5),
        objective_row,
        ft.Container(height=12),
        constraints_container,
        ft.Container(height=12),
        calculate_button,
        ft.Container(height=12),
        result_container,
    )

    rebuild_objective_inputs()
    page.update()


ft.app(main)
