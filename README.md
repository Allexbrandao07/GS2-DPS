# Água do Ar · Differentiated Problem Solving

Global Solution 2026.2 · Modelagem Matemática Avançada · FIAP · Engenharia de Software (1º ano)

Modelagem com limites, derivadas e integrais para dimensionar uma estação da **Água do Ar**, a CitizenTech
do grupo que capta água de beber da umidade do ar com MOFs. A pergunta orientadora é: *quantos quilos de MOF
e qual duração de ciclo uma estação precisa para não ficar sem água no pior mês de São Paulo e do Semiárido?*

## Integrantes

| Nome | RM |
|---|---|
| Állex Brandão | 571383 |
| Enzo Abreu | 572995 |
| Henrique Bueno | 570591 |
| Murilo Gomes | 570941 |

## Arquivos

Tudo fica na pasta [`agua_do_ar_dps/`](agua_do_ar_dps/):

| Arquivo | Conteúdo |
|---|---|
| `agua_do_ar_dps.py` | Código completo, em células `# %%` (abre como notebook no VS Code) |
| `agua_do_ar_dps.ipynb` | O mesmo código em Jupyter Notebook, já executado |
| `relatorio_tecnico.pdf` | Descrição do problema, modelos, deduções e interpretação dos resultados |
| `dados/` | CSVs originais do INMET de 2023: São Paulo (A701), Petrolina (A307) e Curaçá (A448) |
| `figuras/` | Gráficos gerados pelo código |
| `requirements.txt` | Bibliotecas necessárias |

## Como executar

```bash
cd agua_do_ar_dps
pip install -r requirements.txt
python agua_do_ar_dps.py
```

Ou abra `agua_do_ar_dps.ipynb` e execute todas as células. O código não acessa a internet; os dados já estão em
`dados/`.

## Principais resultados

- Para 25 pessoas (50 L/dia) e reservatório de 100 L: cerca de **72 kg de MOF em São Paulo** (pior mês: julho) e
  **60 kg no Semiárido** (pior mês: junho).
- Ciclo ótimo de cerca de 2 h de adsorção e 1 h de dessorção, obtido derivando a taxa de produção e resolvendo
  $e^x = 1 + x + \delta$ pelo método de Newton.
- Abaixo de 1,5 dia de reserva, a massa de MOF necessária cresce muito rápido.

## Uso de Inteligência Artificial

Usamos o Claude (Anthropic), pelo Claude Code no VS Code, como ferramenta de apoio na execução deste
entregável. A condução do trabalho e as decisões foram do grupo, com Állex Brandão à frente.

**Decisões do grupo**

- **Base do trabalho.** O projeto Água do Ar (problema, cidadão atendido, solução e territórios) é nosso e
  orientou toda a modelagem.
- **Direção matemática.** Exigimos que o trabalho aplicasse limites, derivadas, taxas de variação e integrais
  ao projeto, ficando dentro do conteúdo das aulas 12 a 19. Também definimos incluir uma regressão simples,
  treinada com dados públicos.
- **Modelagem.** Avaliamos a proposta de modelos e aprovamos a cadeia final: MOF, ciclo ótimo, simulador
  mensal, dia típico e reservatório.
- **Dados.** Escolhemos usar o INMET, com Petrolina como estação do Semiárido.
- **Conformidade com as aulas.** Pedimos a conferência de cada técnica contra o material da disciplina. Também
  pedimos a troca do que não foi visto em aula (função W de Lambert, bisseção e função de Heaviside) por
  métodos estudados.
- **Formato.** Definimos:
  - um único `.py` e um `.ipynb`, só em Python, sem GeoGebra;
  - código apenas com comentários de guia;
  - explicações concentradas no relatório;
  - identidade visual do projeto no PDF;
  - README na raiz do repositório.

**Onde a IA ajudou**

- Sugerir a formulação dos modelos a partir das nossas diretrizes.
- Implementar o código, baixar e tratar os dados do INMET e gerar os gráficos.
- Redigir o relatório técnico a partir dos resultados.

**Como os resultados foram conferidos.** O próprio código confere cada resultado por um segundo caminho
independente:

- derivada à mão contra o SymPy;
- Newton contra o método de Brent;
- métricas calculadas à mão contra o scikit-learn;
- integral analítica contra o `quad` e as somas de Riemann.

Os números do relatório saem diretamente da execução do código.
