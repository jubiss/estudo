# Aula 1-1
O que é fotografia e introdução histórica sobre fotografia.
Entendimento sobre o que é cor e percepções básicas de cores.

## Tipos de imagens

### Imagens digitais
Caputaradas por um sensor e convertida em um conjunto de pixels, contendo informações de cor ou intensidade.

### Imagens analógicas
Capturada em filme fotográfico por meio de processos químicos.

### Imagens
Associadas a caputra de uma cena em um único quadro

### Vídeos:
Uma sequência por um determinado período de tempo de imagens, criando ilusão do movimento. (Em geral 30fps)

## Camerâs e fotografia perspectiva histórica
Na aula foi falado sobre a história da evolução das câmeras e da fotografia.

## Radiação Eletromagnética
Imagens e suas aplicações podem ser categorizadas de acordo com sua fonte.

Visão computacional pode trabalhar com outros espectros fora do visível.

## Percepção visual de imagens

Cores frias: Associados a cores como azul, violeta e verde.
Cores quentes: Vermelho, laranja e amarelo.
Cores neutras: Acinzentados, marrons e pastéis.

Cores percebidas são relacionadas as luz refletidas.
Luz acromática ou monocromática: luz sem cor.
Fonte de luz pode ser radiância, luminancia e brilho.

## Fotografia
Segundo Michaelis online
"Arte ou processo de reproduzir pela ação da luz ou de qualquer espécia de energia radiante, sobre uma superficie sensibilizada, imagens obtidas mediante uma cêmara escura."

Segundo Websters
"A arte ou processo de produzir imagens pela ação de energia radiante e, especialmente, luz em uma superficie sensível, como filme ou em um sensor óptico"

## Propriedades das cores

Matiz: Cor visível

Saturação: A pureza em termos de mistura de branco

Brilho: Especifica o componente acromático, que é a quantidade de luz emitida ou refletida

## Percepção visual

Conceito de brilho está ligado a percepção visual da quantidade de luz proveniente de uma fonte luminosa.

- O brilho é um elemento crucial ao apresentar resultados que envolvem imagens digitais. (Para o ser humano)

- A sensibilidade do sistema visual humano ao brilho é representada de maneira logarítmica em relação a intensidade de luz que atinge o olho. (Resultados experimentais)

Efeito mach: O olho tende a subestimar ou superestimar a intensidade próxima as transições entre regiões de intensidades diferentes.

## Definição de imagem digital

Descrita por uma função f(x,y)
x,y -> cordenadas de posição
f() -> Intensidade relacionada

## Aquisição de imagens - CCD e CMOS

Sensores utilizados para obter imagens.

CCD consome menor energia
CMOS fotos mais nitidas e menos ruidosas
SCMOS (Tráz o melhor do CCD e do CMOS)

### Modelo de imagem simples
![Modelo de imagem simples](imagens/modelo_de_imagem_simples.png)

L_{min} e L_{max} dependem do sensor.

### Amostragem e quantificação

Vai ser quantizado o sinal para uma transformação do sinal de analógico para digital.

É obtido uma amostragem desse sinal.

### Representação da imagem

São representadas em um plano cartesiano bi-dimensional, em que cada ponto possui o valor da intensidade do pixel.

- Saturação: É o valor máximo no qual todos os valores de intensidade são cortados em uma certa região da imagem.
A região saturada tem um nível de intensidade alto e constante.

- Ruído visível: Padrão de textura granulada na imagem.

- Faixa dinâmica: Razão entre intensidade máxima mensuável e o nível mínimo detectável de intensidade no sistema.
Regra geral, o limite superior é determinado pela saturação e o inferior pelo ruído.

### Memória ocupada para representar a imagem
![Memoria Ocupada por tipo de imagem](imagens/memoria_ocupada_por_tamanho_de_imagem.png)

### Efeito do undersampling sobre solução da imagem!

![Efeito de undersampling sobre imagem](imagens/efeito_de_undersampling_sobre_imagem.png)

# Aula 1-2

## Transformações básicas

### Negativo:

Em que f(x,y) = Intensidade do pixel
L é a intensidade máxima do pixel (255)
g(x,y) = Intensidade do após transformação
g(x,y) = (L - 1) - f(x,y)

### Translação
Importante assegurar que o deslocamento não seja maior que a dimensão da imagem.

Translada a imagem
g(x, y) = g(x + a, y+ b)

### Resize
Redimensionamento da imagem

### Reflexão

Refletir a imagem em um dado eixo.

### Reflexão + Translação

Reflete e translada uma image

### Rotação

M = [cos(theta), -sin(theta)
    sin(theta), cos(theta)]

Rotação em escala olhar no openCV

### Rotação geral (Warp affine)

- Transformação afim: Preserva a colinearidade e o paralelismo bem como a razão das distâncias entre os pontos
Ex: O ponto médio permanece os mesmos, mas não preserva necessáriamente distâncias e ângulos.

## Transformações básicas em nível de cinza

g(x, y) = T[f(x,y)]

T é uma transformação.

A forma mais simples de transformação envolve um pixel.
Utilizar uma T(r), transformação logística. Em que **r** é um Threshold.
### Transformação linear

s = T(r) = (L - 1) - r

### Transformação logaritmica

s = c log(1+r) r>=0

c é definido de forma que s_max = 255

c = 255/log(1+R) para R = r_max

### Correção gama
Correção gama exemplo de uso:
Clarear uma imagem
Escurecer uma imagem

s = c r^(gama)

### Estiramente de contraste

r1 = r2 e s1 = 0 e s2 = L-1
Utiliza uma função de threshold.

O alargamento do contraste é obtido através de

(r1, s1) = (r_min, 0) e (r2, s2)= (r_max, L-1)

r_min e r_max são os máximos minimos dos pixels.

### Fatiamento de níveis de cinza

Faixas de intensidade de pixel. (Piecewise Function)
Fatiamento por planos de bits.

## Histograma

Gráfico de barras que mostra distribuição de frequÊncia
Ilustra distribuição de uma amostra de dados.

### Histograma aplicados a imagens

Conseguimos utilizar histogramas para entender a distribuição das intensidades de cada pixel

### Histograma - Equalização

Equalização é uma forma de transformação, utilizando um histograma de forma a redistribuir a distribuição dos pixels.

Nem toda equalização vai melhorar a imagem.

Passo a Passo
1- Normalize a distribuição de r_k
2- Cálcule a distribuição acumulada
3- Multiplicar os valores da distribuição acumulada pelo Pixel de intensidade máxima (L-1), e arredondar o número final
4- Mapeamento dos bins (Valor de r_k, com valor final obtido)

### Histograma - Realce Local

Fazer a equalização usando uma vizinhança 7 x 7 sobre cada pixel

### Histograma - Realce de Imagem baseado em estatísticas locais

Utiliza o momento e desvio padrão para fazer a equalização

### Histograma - Fotografias em diferentes celulares

Diferentes celulares vão ter histogramas diferentes

### Realce usando operações aritmétricas/Lógicas

Realiza o realce só em parte da imagem

### Realce usando operações aritmétrica/Lógicas Subtração de imagens

Uso de subtração de imagens para realce

# Aula 1-3

Continuação da aula 2 de realce

### Realce usando operações aritmétrica/Lógicas médias de imagens

g(x, y) = f(x, y) + n(x, y)

g_medio(x, y) = (sum_i^k g_i(x, y))/k

## Fundamentos de Filtros espaciais

Adiciona pesos ao longo de uma máscara (ex: Máscara 3x3)

É mostrado o fundamento e funcionamento das máscaras em tremos matemáticos

Para realização do filtro, se multiplica os pesos da mascara pelas intensidades da imagem, é gerado um valor resultando no pixel central.

Após a convolução perdemos as bordas da imagem.

### Filtros espaciais de suavização

#### Filtro utilizando a média
Utilizado para borramento e redução do ruido,
Borramento: Remoção de pequenos detalhes de uma imagem antes da extração de objetos.
Redução de ruído: Obtida com borramento com filtro linear ou filtragem não linear.

Tipos de filtragem espacial passa-baixa:
- Utiliza a média dos vizinhos
- Gera uma imagem borrada
- Pode reduzir o ruído

Durante a filtragem é feito uma filtagem normalizada.

 #### Filtro utilizando a mediana

 Mesma coisa que a média mas usando a mediana

- Melhor na redução de ruído

#### Filtro espacial de aguçamente

- Objetivo: Enfatizar detalhes finos ou realçar detalhes tenham sido borrados
- Média tende a borrar os detalhes, diferenciação faz o oposto ressaltando as variações e contornos

- Borramento - Integração
- Aguçamento - Diferenciação

Tipos de filtros de aguçamento

1- Espacial passa-altas básico
2- Por alto reforço
3- Derivadas

#### Filtragem espacial psasa-altas básica

Formas de ver:
- Seção transversal ao filtro no domínio da frequência

Para frequências menores temos a atenuação de sinal
Frequências maiores são passadas

- Seção transversal do filtro no domínio espacial
A função fica cada vez mais próxima do pixel central

- Positivo no centro e negativo nos arredores
#### Filtragem espacial por alto reforço

- Desaguçamento ou mascara de unsharp
- Filtro passa-alta = original - passa-baixa (desaguçamento)

1- É realizado um borramento sobre a imagem
2- É feita a subtração dessa imagem suavizada na original
3- Com a adição do resultado final a parte de alta frequência é intensificada

A=1 Filtro padrão passa alta

A> 1 Filtro Alto reforço

Filtro passa alta em que o centro soma por um valor A

## Filtros espaciais

### Filtros por derivadas

Como diferencias uma imagem digital

Op1: Fazer a imagem se tornar contínua

Op2: Utilizar a derivação discreta

### Detecção de boarda

1- Se detecta uma borda a partir dos máximos ou mínimos de primeira derivada

2- Encontrar o valor da segunda derivada

Pode ser feito com derivadas normais ou parciais

### Filtro Laplaciano

Filtro invariante em rotação (Isotrópicos)

Filtros laplacianos com 4 vizinhanças

Filtros laplacianos com 8 vizinhanças

### Filtro Gradiente

- Magnitudo é variante com rotação (Não isotropica)
- Implementação computacional é não trivial
- É comum a aproximação da magnitude do gradiente usando valores absolutos

Gradiente diferença entre os vizinhos

Gradiente de Roberts, diferença entre vizinhos cruzados

Filtro de Sobel
Serve para detectar contorno

Filtro de Prewitt

Observa as derivadas horizontais e verticais


# Aula 1-4
Aula sobre transformações no dominio da frequência

Série de Fourrier: Utilizado para funções periódicas

Transformada de Fourier: Funções que não são periódicas, mas que possui uma área abaixo da curva um valor finito

Séries e transfomadas de Fourier conseguem ser invertidas

## Fluxo:
Pré processamento -> Transformada de Fourrier -> Filtro no dominio da frequência -> Transformada inversa -> Pós processamento

## Procedimento:
![procedimento filtragem na frequência](<imagens/procedimento filtragem na frequência.png>)

## Tipos de filtros:
Passa baixa (suavização, borramento)
    Preserva baixa frequência
    Suprime as altas frequências
Passa alta (realce das bordas, aguçamento)
    Preserva altas frequência
    Suprime baixas frequências
Passa faixa (Restauração de imagens)
    Preserva frequência especificas
    Suprime o resto


## Filtro básico:

Filtro tipo Notch ou rejeita faixa
0 se satisfaz condição ou 1

### Filtro Ideal (Passa-baixa) LPF

É definido 1 se for maior que um valor 0 se contrario

### Filtros utilizados para cortes suaves
Filttro de Butterworth ou filtro gaussiano passa-baixa

### Filtragem por alto reforço

passa-altas = normal - passa baixa
É possível usar o laplaciano para fazer o desaguçamento (Unsharp Masking)

### Filtragem com enfase em alta frequência (HFE)

### Filtragem Homomórfica


