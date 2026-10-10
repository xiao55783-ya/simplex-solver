import flet as ft 
def main(page: ft.Page):
    page.title="simplexe solver"
    page.vertical_alignment=ft.MainAxisAlignment.START
    page.horizontal_alignment=ft.CrossAxisAlignment.CENTER
    page.scroll=ft.ScrollMode.AUTO
    
    
    title_text=ft.Text("Objective Function",size=18,weight=ft.FontWeight.BOLD)
    
    
    x1_input=ft.TextField(label="x1 coef",width=150)
    x2_input=ft.TextField(label="x2 coef",width=150)
    inputs_row=ft.Row(
        [x1_input,x2_input],
        alignment=ft.MainAxisAlignment.CENTER,
        spacing=10
        )
    
    selected_objective="Max"
    def Objective_clicked(e):
        nonlocal selected_objective
        if e.control.text == "Max":
            selected_objective="Max"
            Max_btn.bgcolor=ft.Colors.BLUE
            Max_btn.color=ft.Colors.WHITE
            Min_btn.bgcolor=ft.Colors.GREY_200
            Min_btn.color=ft.Colors.BLACK
        else:
            selected_objective="Min"
            Min_btn.bgcolor=ft.Colors.BLUE
            Min_btn.color=ft.Colors.WHITE
            Max_btn.bgcolor=ft.Colors.GREY_200
            Max_btn.color=ft.Colors.BLACK
        page.update()    
    
    
    Max_btn=ft.ElevatedButton("Max",bgcolor=ft.Colors.BLUE,color=ft.Colors.WHITE,width=90,on_click=Objective_clicked)
    Min_btn = ft.ElevatedButton("Min", bgcolor=ft.Colors.GREY_200, color=ft.Colors.BLACK, width=90, on_click=Objective_clicked)
    choise_row=ft.Row(
        [Max_btn,Min_btn],
        alignment=ft.MainAxisAlignment.CENTER,
        spacing=10
        )
    
    
    
    x1_input_cont=ft.TextField(label="x1 coef",width=60,height=40,text_size=12)
    x2_input_cont=ft.TextField(label="x2 coef",width=60,height=40,text_size=12)
    
    sign_dropdown=ft.Dropdown(
        options=[
            ft.dropdown.Option("<="),
            ft.dropdown.Option(">="),
            ft.dropdown.Option("<"),
            ft.dropdown.Option(">"),
            ft.dropdown.Option("="),
            ],
        value="<=",
        width=55,
        height=40,
        text_size=12
        )
    
    rhs_input=ft.TextField(label="RHS",width=55,height=40,text_size=12)
    
    constraint_dialog=ft.AlertDialog(
        title=ft.Text("Add new constraint",size=16,weight=ft.FontWeight.BOLD),
        content=ft.Row(
            [x1_input_cont,x2_input_cont,sign_dropdown,rhs_input],
            alignment=ft.MainAxisAlignment.CENTER,
            spacing=3
            ),
        actions=[
            ft.TextButton("Cansel",on_click=lambda e: close_dlg(e)),
            ft.TextButton("Add",on_click=lambda e: save_constraint(e))
            ],
        actions_alignment=ft.MainAxisAlignment.END
        )
    def open_dlg(e):
        if constraint_dialog not in page.overlay:
            page.overlay.append(constraint_dialog)
        constraint_dialog.open=True
        page.update()
    def save_constraint(e):
        c1=x1_input_cont.value.strip() if x1_input_cont.value else "0"
        c2=x2_input_cont.value.strip() if x2_input_cont.value else "0"
        sign=sign_dropdown.value
        rhs=rhs_input.value.strip() if rhs_input.value else "0"
        new_constraint = {"c1": c1, "c2": c2, "sign": sign, "rhs": rhs}
        constraints_list.append(new_constraint)
        x1_input_cont.value=""
        x2_input_cont.value=""
        rhs_input.value=""
        sign_dropdown.value="<="
        constraint_dialog.open=False
        update_constraints_ui()
        page.update()
    def close_dlg(e):
        constraint_dialog.open=False
        page.update()    
    
    
    constraints_list=[]
    constraints_title=ft.Text("Constraints",size=18,weight=ft.FontWeight.BOLD)
    constaints_column=ft.Column(spacing=5)
    def update_constraints_ui():
        constaints_column.controls.clear()
        for i,constraint in enumerate(constraints_list):
            def remove_constraint(e,idx=i):
                constraints_list.pop(idx)
                update_constraints_ui()
                page.update()
            c_text = f"({i+1}) {constraint['c1']}x1 + {constraint['c2']}x2 {constraint['sign']} {constraint['rhs']}"    
            row_item=ft.Row(
                [
                    ft.Text(c_text,size=14,weight=ft.FontWeight.W_500),
                    ft.IconButton(icon=ft.Icons.DELETE_OUTLINE,icon_color=ft.Colors.RED_400,
                                  on_click=remove_constraint
                                  )
                    ],
                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                width=290
                )
            constaints_column.controls.append(row_item)
    update_constraints_ui()        
    
    add_contraint_btn=ft.TextButton("Add Constraint +",icon=ft.Icons.ADD,on_click=open_dlg)
    constraint_container=ft.Container(
        content=ft.Column(
            [constraints_title,constaints_column,add_contraint_btn],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=5
            ),
        padding=10,
        border=ft.border.all(1,ft.Colors.GREY_300),
        border_radius=10,
        width=320
        )
    
    
    result_display=ft.Text("Z = -- ",size=20,weight=ft.FontWeight.BOLD,color=ft.Colors.GREEN_700)
    details_desplay=ft.Text("x1= -- , x2 = -- ",size=14)
    
    
    def solve_simplex(c1, c2, constraints, obj_type):
        mult = 1.0
        if obj_type == "Min":
            c1 = -c1
            c2 = -c2
            mult = -1.0

        m = len(constraints)
        num_vars = 2
        cols = num_vars + m + 1
        rows = m + 1

        table = [[0.0 for _ in range(cols)] for _ in range(rows)]

        for i, constraint in enumerate(constraints):
            try:
                table[i][0] = float(constraint['c1'])
                table[i][1] = float(constraint['c2'])
                table[i][num_vars + i] = 1.0
                table[i][-1] = float(constraint['rhs'])
            except Exception:
                pass

        table[m][0] = -c1
        table[m][1] = -c2

        max_iterations = 100
        iteration = 0

        while iteration < max_iterations:
            col_pivot = -1
            min_val = 0.0
            for j in range(cols - 1):
                if table[m][j] < min_val:
                    min_val = table[m][j]
                    col_pivot = j

            if col_pivot == -1:
                break

            row_pivot = -1
            min_ratio = float('inf')
            for i in range(m):
                if table[i][col_pivot] > 0:
                    ratio = table[i][-1] / table[i][col_pivot]
                    if ratio < min_ratio:
                        min_ratio = ratio
                        row_pivot = i

            if row_pivot == -1:
                return None, None, "Unbounded"

            pivot_element = table[row_pivot][col_pivot]
            for j in range(cols):
                table[row_pivot][j] /= pivot_element

            for i in range(rows):
                if i != row_pivot:
                    factor = table[i][col_pivot]
                    for j in range(cols):
                        table[i][j] -= factor * table[row_pivot][j]

            iteration += 1

        x1_val = 0.0
        x2_val = 0.0

        for j in range(num_vars):
            col_column = [table[i][j] for i in range(rows)]
            if col_column.count(1.0) == 1 and col_column.count(0.0) == rows - 1:
                idx_one = col_column.index(1.0)
                if j == 0:
                    x1_val = table[idx_one][-1]
                elif j == 1:
                    x2_val = table[idx_one][-1]

        z_val = table[m][-1] * mult
        return x1_val, x2_val, z_val
    
    
    def calculate_clicked(e):
        try:
            c1=float(x1_input.value) if x1_input.value else 0.0
            c2=float(x2_input.value) if x2_input.value else 0.0
            
            details_desplay.value=f"x1 = {c1} , x2 = {c2}"
            page.update()
        except ValueError:
            details_desplay.value = "Please enter valid numbers for objective function!"
            page.update()
            return
            
        if len(constraints_list) == 0:
            result_display.value = "Error!"
            details_desplay.value = "Please add at least one constraint!"
            page.update()
            return

        obj_type = selected_objective
        
      
        x1_res, x2_res, z_res = solve_simplex(c1, c2, constraints_list, obj_type)
        
        if z_res == "Unbounded":
            result_display.value = "Unbounded!"
            details_desplay.value = "The solution is unbounded."
        else:
            result_display.value = f"{obj_type} Z = {z_res:.2f}"
            details_desplay.value = f"x1 = {x1_res:.2f} , x2 = {x2_res:.2f}"
            
        page.update()
        
        
    calc_button = ft.ElevatedButton(text="calculate",bgcolor=ft.Colors.BLUE_600,color=ft.Colors.WHITE,width=200, on_click=calculate_clicked)    
    
    
    result_container=ft.Container(
        content=ft.Column(
            [
                ft.Text("Result",size=14,weight=ft.FontWeight.BOLD),
                result_display,
                details_desplay,
                ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=5
            ),
        padding=15,
        border=ft.border.all(1,ft.Colors.GREEN_300),
        border_radius=10,
        bgcolor=ft.Colors.GREEN_50,
        width=320
        )
    
    
    page.add(title_text,
             inputs_row,
             choise_row,
             ft.Divider(height=20,color=ft.Colors.TRANSPARENT),
             constraint_container,
             ft.Divider(height=20,color=ft.Colors.TRANSPARENT),
             calc_button,
             ft.Divider(height=20,color=ft.Colors.TRANSPARENT),
             result_container
             )
    
    
    page.update()
    
    

    
    
