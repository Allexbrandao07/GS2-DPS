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
| Erick Nathan Capitó | 573635 |
| Henrique Bueno | 570591 |
| Murilo Gomes | 570941 |

## Arquivos

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

Usamos o Claude (Anthropic), pelo Claude Code no VS Code, como ferramenta de apoio neste entregável.

**Partes do grupo**

- O projeto Água do Ar (problema, cidadão atendido, solução e territórios), que é a base de toda a modelagem.
- A direção do trabalho: aplicar limites, derivadas e integrais ao projeto, seguir o material das aulas 12 a 19
  e incluir uma regressão simples treinada com dados públicos.
- As decisões de escopo e formato:
  - usar Petrolina para o Semiárido;
  - entregar um único `.py` e um `.ipynb`, só em Python;
  - manter o código enxuto e as explicações no relatório;
  - aplicar a identidade visual do projeto;
  - substituir ferramentas que não foram vistas em aula.

**Partes da IA**

- Propor a cadeia de modelos (isoterma, ciclo ótimo, simulador mensal, regressão harmônica e reservatório), a
  partir do documento do projeto e do material das aulas.
- Escrever o código, baixar e tratar os dados do INMET e gerar os gráficos.
- Redigir o relatório técnico.

**Como os resultados foram conferidos.** O próprio código confere cada resultado por um segundo caminho
independente:

- derivada à mão contra o SymPy;
- Newton contra o método de Brent;
- métricas calculadas à mão contra o scikit-learn;
- integral analítica contra o `quad` e as somas de Riemann.

Os números do relatório saem diretamente da execução do código.
