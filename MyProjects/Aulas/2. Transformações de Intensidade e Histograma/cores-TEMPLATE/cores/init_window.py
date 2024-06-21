import tkinter as tk
from tkinter import filedialog
import tkinter.font
from PIL import Image, ImageTk
import os
import filtering
import color

class Window(tk.Tk):
  
    def __init__(self):
        super().__init__()    
        self.image = None 
        self.image_copy = None
        self.resized_image = None
        self.tkimage = None
        self.reset_button = None 
        self.create_main_window()
        self.create_control_frame()
        self.create_image_canvas()       
        self.create_menubar()  

    # Cria a janela principal
    def create_main_window(self):
        self.title('Filtragem de Imagem')
        self.width= self.winfo_screenwidth()               
        self.height= self.winfo_screenheight()            
        self.geometry("%dx%d" % (self.width*.8, self.height*.8))
        self.state('zoomed')
    
    # Cria o frame onde serao posicionados controles como botao, spinner, slide, etc
    def create_control_frame(self):
        self.controls = tk.Frame(self)
        self.controls.pack(side=tk.TOP,expand=True,pady=(0,10))

     # Limpa o frame de controle
    def clear_control_frame(self):
        for widget in self.controls.winfo_children():
            widget.destroy()

    # Cria a barra de menu
    def create_menubar(self): 
        # Cria a barra de menu
        self.menu_bar = tk.Menu(self, tearoff="off")
        self.config(menu=self.menu_bar)
        # Define da estrutura da barra de menu utilizando um dicionario aninhado
        menus = {
            'Arquivo': {
                'Abrir...': self.load_image,
                'Sair': self.quit
            },
            'Imagem': {
                'Redimensionar': self.create_resize_image_controls,
                'Escala de cinza': self.create_color_controls(color.grayscale_image)
            },
            'Filtros': {
                'Suavização': {
                    'Média': self.create_filter_controls(filtering.average_filter),
                    'Gaussiano': self.create_filter_controls(filtering.gaussian_filter),
                    'Mediana': self.create_filter_controls(filtering.median_filter)
                },
                'Aguçamento': {
                    'Sobel': self.create_filter_controls(filtering.sobel_filter),
                    'Laplaciano': self.create_filter_controls(filtering.laplacian_filter),
                    'High-boost': self.create_filter_controls(filtering.highboost_filter)
                },
                'Ruído': {
                    'Sal e Pimenta': self.create_filter_controls(filtering.salt_and_pepper_noise)
                }
            },
            'Cores': {
                'Negativo': self.create_color_controls(color.negative_image),
                'Brilho e Contraste': {
                    'Transformação Logarítmica':self.create_color_controls(color.log_transform),
                    'Correção Gamma':self.create_gamma_correction_controls,
                    'Ajuste de Contraste':self.create_contrast_stretch_controls,
                    'Equalização de Histograma':self.create_color_controls(color.histogram_equalization),
                },
                'Histograma': self.create_histogram_controls
            }
        } 
        # Itera sobre o dicionário aninhado para criar de cada item de menu
        for menu_name, menu_items in menus.items():
            menu = tk.Menu(self.menu_bar, tearoff="off")
            self.menu_bar.add_cascade(label=menu_name, menu=menu)
            for item_name, command in menu_items.items():
                # Se command for um outro dicionario, cria-se um submenu
                if isinstance(command, dict):
                    submenu = tk.Menu(menu, tearoff="off")
                    menu.add_cascade(label=item_name, menu=submenu)
                    for subitem_name, subcommand in command.items():
                        submenu.add_command(label=subitem_name, command=subcommand)
                # Senao, cria um item de menu
                else:
                    menu.add_command(label=item_name, command=command)
        # Desabilita os menus enquanto a imagem nao for carregada/aberta
        self.disable_menus()
  
    # Desabilita menus
    def disable_menus(self):
        self.menu_bar.entryconfig('Filtros', state='disabled')
        self.menu_bar.entryconfig('Imagem', state='disabled')
        self.menu_bar.entryconfig('Cores', state='disabled')

    # Habilita menus
    def enable_menus(self):
        self.menu_bar.entryconfig('Filtros', state='active')
        self.menu_bar.entryconfig('Imagem', state='active')
        self.menu_bar.entryconfig('Cores', state='active')

    # Cria o canvas para exibição da imagem
    def create_image_canvas(self):
        image_frame = tk.Frame(self)
        image_frame.pack()
        yscrollbar = tk.Scrollbar(image_frame, orient = tk.VERTICAL)
        yscrollbar.pack(side = tk.RIGHT, fill = tk.Y)
        xscrollbar = tk.Scrollbar(image_frame, orient = tk.HORIZONTAL)
        xscrollbar.pack(side = tk.BOTTOM, fill = tk.X)
        self.image_canvas = tk.Canvas(image_frame, 
                                      width = self.width*.8, 
                                      height = self.height*.75,
                                      xscrollcommand = xscrollbar.set, 
                                      yscrollcommand = yscrollbar.set)
        self.image_canvas.pack()
        yscrollbar.config(command = self.image_canvas.yview)
        xscrollbar.config(command = self.image_canvas.xview)

    # Carrega a imagem para exibicao no canvas
    def load_image(self):
        self.clear_control_frame()
        filename = filedialog.askopenfilename(initialdir=os.getcwd())
        if filename:
            img = filtering.read_image(filename=filename)
            if (img is not None):
                self.image = img.copy()
                self.image_copy = img.copy()
                self.resized_image = img.copy() 
                self.display_image()
                self.enable_menus()
    
    # Exibe a imagem no canvas
    def display_image(self):
        img = Image.fromarray(self.image)
        self.tkimage = ImageTk.PhotoImage(img)
        width,height = img.size
        self.image_canvas.config(scrollregion=(0,0,width,height))
        self.image_canvas.create_image(0,0,anchor="nw",image=self.tkimage)

    # Exibe os valores dos pixels
    def display_pixel_values(self):
        self.clear_control_frame()
        filtering.show_pixel_values(self.image)

    # Cria os controles para redimensionar a imagem
    def create_resize_image_controls(self):
        self.clear_control_frame()
        width_label = tk.Label(self.controls,text='Largura: ', height=4)
        width_label.pack(side=tk.LEFT)
        font = tkinter.font.Font(family='Helvetica', size=12, weight='bold')
        var = tk.IntVar()
        var.set(25) #default value
        self.width_spin = tk.Spinbox(self.controls, from_=1, to=200, textvariable=var, increment=1, font=font, width=4)
        self.width_spin.pack(side=tk.LEFT)
        perc_label1 = tk.Label(self.controls,text=' %', height=4)
        perc_label1.pack(side=tk.LEFT)
        heigth_label = tk.Label(self.controls,text='Altura: ', height=4)
        heigth_label.pack(side=tk.LEFT)
        self.height_spin = tk.Spinbox(self.controls, from_=1, to=200, textvariable=var, increment=1, font=font, width=4)
        self.height_spin.pack(side=tk.LEFT)
        perc_label2 = tk.Label(self.controls,text=' %', height=4)
        perc_label2.pack(side=tk.LEFT)
        button = tk.Button(self.controls, text='Aplicar', command=self.apply_resize_image,width=20,height=1)
        button.pack(side=tk.LEFT) 
        self.reset_button = tk.Button(self.controls, text='Reset', command=self.reset_size,width=20,height=1)      

    # volta a imagem para o estado inicial logo após o redimensionamento
    def reset_image(self):
        self.image = self.resized_image.copy()
        self.display_image()

    # Redefine o tamanho da imagem para o tamanho original
    def reset_size(self):
        self.image = self.image_copy.copy()
        self.resized_image = self.image_copy.copy()
        self.display_image()

    # Aplica o redimensionamento da imagem
    def apply_resize_image(self):
        width = int(self.width_spin.get())
        height = int(self.height_spin.get())
        self.image = filtering.resize_image(self.image,width,height)
        self.resized_image = self.image.copy() #stores a copy of the resized image for reset option
        self.reset_button.pack(side=tk.LEFT,padx=5)
        self.display_image()    

    # Cria os controles para os filtros
    def create_filter_controls(self, filter_function):
        def create_controls():
            self.clear_control_frame()
            font = tkinter.font.Font(family='Helvetica', size=12, weight='bold')
            
            if filter_function in [filtering.average_filter, filtering.gaussian_filter, filtering.median_filter]:
                var = tk.IntVar()
                var.set(3) 
                kernel_label = tk.Label(self.controls, text='Tamanho do kernel: ', height=4)
                kernel_label.pack(side=tk.LEFT)
                self.kernel_spin = tk.Spinbox(self.controls, from_=3, to=31, textvariable=var, increment=2, font=font, width=2)
                self.kernel_spin.pack(side=tk.LEFT)
                slider = tk.Scale(self.controls, from_=3, to=31, variable=var, tickinterval=2, resolution=2, length=300, orient="horizontal")
                slider.pack(side=tk.LEFT)
            elif filter_function == filtering.highboost_filter:
                font = tkinter.font.Font(family='Helvetica', size=12, weight='bold')
                var = tk.DoubleVar()
                var.set(1) 
                factor_label = tk.Label(self.controls,text='Fator de amplificação: ', height=4)
                factor_label.pack(side=tk.LEFT)
                self.boost_spin = tk.Spinbox(self.controls, from_=1, to=3, textvariable=var, increment=0.1, font=font, width=3)
                self.boost_spin.pack(side=tk.LEFT)
                slider = tk.Scale(self.controls, from_=1, to=3, variable=var, tickinterval=1, resolution=0.1, length=300, orient="horizontal")
                slider.pack(side=tk.LEFT)
            
            button = tk.Button(self.controls, text='Aplicar', command=apply_filter, width=20, height=1)
            button.pack(side=tk.LEFT) 
            self.remove_button = tk.Button(self.controls, text='Remover Filtros', command=self.remove_filters, width=20, height=1)
        
        # Aplica o filtro de acordo com o parametro filter_function
        def apply_filter():
            if filter_function in [filtering.average_filter, filtering.gaussian_filter, filtering.median_filter]:
                kernel_size = int(self.kernel_spin.get())
                self.image = filter_function(self.image, kernel_size)
            elif filter_function == filtering.highboost_filter:
                boost = float(self.boost_spin.get())
                self.image = filter_function(self.image, boost)
            else:
                self.image = filter_function(self.image)
            self.remove_button.pack(side=tk.LEFT, padx=5)
            self.display_image()

        return create_controls

    # Remove todos os filtros aplicados
    def remove_filters(self):
        self.image = self.resized_image.copy()
        self.display_image()    

    def create_color_controls(self,transform_function):
        def create_controls():
            self.clear_control_frame()
            button_apply = tk.Button(self.controls, text='Aplicar', command=apply_image_transform, width=20, height=1)
            button_apply.pack(side=tk.LEFT) 
            self.reset_button = tk.Button(self.controls, text='Reset', command=self.reset_image, width=20, height=1)

        def apply_image_transform():
            self.image = transform_function(self.image)
            self.reset_button.pack(side=tk.LEFT, padx=5)
            self.display_image()
        
        return create_controls
    
    def create_gamma_correction_controls(self):
        self.clear_control_frame()
        font = tkinter.font.Font(family='Helvetica', size=12, weight='bold')
        const_var = tk.DoubleVar()
        const_var.set(1) #default value
        gamma_label = tk.Label(self.controls,text='Constante C: ', height=4)
        gamma_label.pack(side=tk.LEFT)
        self.gamma_spin = tk.Spinbox(self.controls, from_=0.1, to=3, textvariable=const_var, increment=0.1, font=font, width=3)
        self.gamma_spin.pack(side=tk.LEFT)
        slider = tk.Scale(self.controls,from_=0.1, to=3, variable=const_var,tickinterval=1, resolution=0.1,length= 300, orient="horizontal")
        slider.pack(side=tk.LEFT)
        button = tk.Button(self.controls, text='Aplicar', command=self.apply_gamma_correction,width=20,height=1)
        button.pack(side=tk.LEFT) 
        self.reset_button = tk.Button(self.controls, text='Reset', command=self.reset_image,width=20,height=1)

    def create_contrast_stretch_controls(self):
        self.clear_control_frame()
        font = tkinter.font.Font(family='Helvetica', size=12, weight='bold')
        const1_var = tk.IntVar()
        const1_var.set(255) #default value
        max_label = tk.Label(self.controls,text='Maior nível de intensidade: ', height=4)
        max_label.pack(side=tk.LEFT)
        self.max_spin = tk.Spinbox(self.controls, from_=128, to=255, textvariable=const1_var, increment=1, font=font, width=3)
        self.max_spin.pack(side=tk.LEFT)
        slider1 = tk.Scale(self.controls,from_=128, to=255, variable=const1_var,tickinterval=10, resolution=1,length= 300, orient="horizontal")
        slider1.pack(side=tk.LEFT)
        const2_var = tk.IntVar()
        const2_var.set(0) #default value
        min_label = tk.Label(self.controls,text='Menor nível de intensidade: ', height=4)
        min_label.pack(side=tk.LEFT)
        self.min_spin = tk.Spinbox(self.controls, from_=0, to=127, textvariable=const2_var, increment=1, font=font, width=3)
        self.min_spin.pack(side=tk.LEFT)
        slider2 = tk.Scale(self.controls,from_=0, to=127, variable=const2_var,tickinterval=10, resolution=1,length= 300, orient="horizontal")
        slider2.pack(side=tk.LEFT)
        button = tk.Button(self.controls, text='Aplicar', command=self.apply_contrast_stretch,width=20,height=1)
        button.pack(side=tk.LEFT) 
        self.reset_button = tk.Button(self.controls, text='Reset', command=self.reset_image,width=20,height=1)

    def create_histogram_controls(self):
        self.clear_control_frame()
        color.show_histogram(self.image)

    def apply_gamma_correction(self):
        gamma = float(self.gamma_spin.get())
        self.image = color.gamma_correction(self.image,gamma)
        self.reset_button.pack(side=tk.LEFT,padx=5)
        self.display_image() 

    def apply_contrast_stretch(self):
        max = int(self.max_spin.get())
        min = int(self.min_spin.get())
        self.image = color.contrast_stretch(self.image,max,min)
        self.reset_button.pack(side=tk.LEFT,padx=5)
        self.display_image() 

if __name__== '__main__':
    app=Window()
    app.mainloop()