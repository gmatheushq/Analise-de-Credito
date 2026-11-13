# Relatório da Etapa 1: Tratamento e Exploração dos Dados

**Projeto:** Análise de Crédito: Tomada de Decisão e Incerteza Inerente
**Notebook:** `analise_credito_TRATAMENTO.ipynb`
**Base:** `acquisition_train.csv` (45.000 solicitações de crédito, 43 colunas)
**Repositório:** github.com/gmatheushq/CreditRisk

---

## 1. Objetivo da etapa

Conhecer a base antes de modelar. A etapa responde a quatro perguntas:

1. O que existe na base (estrutura, tipos, significado das variáveis)?
2. Qual é a qualidade dos dados (ausências, duplicatas, constantes, outliers)?
3. Como a variável-alvo (inadimplência) se comporta?
4. O que precisa ser corrigido, removido ou transformado antes da modelagem?

Em análise de crédito, erros de dados viram erros de decisão: um limite concedido a quem não deveria, ou negado a quem pagaria. Por isso o diagnóstico vem antes de qualquer modelo.

---

## 2. Estrutura do notebook e justificativa de cada bloco

### 2.1 Bibliotecas e carregamento (seções 1.1 a 1.3)
- **O que faz:** importa `pandas`, `numpy`, `matplotlib` e `seaborn`, carrega o CSV direto do GitHub e guarda uma cópia (`df_original`).
- **Por quê:** a cópia garante que sempre dá para comparar o dado tratado com o bruto (seção 7.10) e voltar atrás sem recarregar.
- **Observação:** a célula 3 ainda monta o Google Drive e define um caminho para `credit_risk_dataset.csv`, que não é usado. É resíduo e deve ser removido.

### 2.2 Estrutura e tipos (seções 1.4 e 1.5)
- **O que faz:** `df.info()` e separação entre variáveis numéricas (22) e categóricas (21).
- **Por quê:** o tipo da variável define quais estatísticas, gráficos e transformações fazem sentido.

### 2.3 Dicionário de dados (seções 2.1 e 2.2)
- **O que faz:** monta uma tabela com tipo, preenchimento, percentual ausente e valores distintos de cada coluna, mais uma descrição semântica de cada uma.
- **Por quê:** a base não vem com documentação. Sem dicionário não há como decidir o que manter. As descrições são interpretações a partir do nome das colunas e devem ser validadas com quem conhece a origem dos dados.

### 2.4 Qualidade dos dados (seção 3)
| Verificação | Resultado | Leitura |
|---|---|---|
| Duplicatas | 0 linhas duplicadas, 0 `ids` repetidos | Base íntegra quanto à unicidade |
| Ausentes | Variam de 0% a 96,6% | Principal problema da base |
| Constantes | `channel` e `external_data_provider_credit_checks_last_2_year` têm 1 valor | Sem poder discriminatório |
| Quase constantes | `n_defaulted_loans` (99,6% numa só categoria) | Pouca informação |

**Classificação de ausência** (célula 21): sem ausência, baixa (até 5%), moderada (5% a 20%), alta (20% a 50%) e muito alta (acima de 50%). Isso permite definir estratégias diferentes por faixa, em vez de tratar tudo igual.

Principais ausências:

| Variável | % ausente |
|---|---|
| `target_fraud` | 96,6% |
| `last_borrowed_in_months` e `last_amount_borrowed` | 66,6% |
| `ok_since` | 59,0% |
| `external_data_provider_credit_checks_last_2_year` | 50,3% |
| `external_data_provider_credit_checks_last_year` | 33,6% |
| `credit_limit` | 30,7% |

### 2.5 Estatísticas descritivas (seção 4)
- **O que faz:** média, mediana, variância, assimetria e curtose das numéricas; histogramas, boxplots e contagens das categóricas.
- **Por quê:** a assimetria e a curtose mostram quais variáveis exigem transformação. Destaques: `income` (assimetria 23,8), `n_defaulted_loans` (23,3) e `external_data_provider_email_seen_before` (-7,7).

### 2.6 Outliers (seções 4.3 e 4.4)
- **O que faz:** método IQR (1,5 × amplitude interquartil) por variável numérica, e inspeção dos 10 maiores valores de renda, limite, valor emprestado e taxa de risco.
- **Por quê:** valores extremos distorcem médias, escalonamento e modelos lineares. Em crédito, porém, um valor extremo pode ser legítimo (renda alta real) ou erro de digitação, e a decisão depende de investigação.
- **Achados:** `income` chega a cerca de 5 milhões; `reported_income` contém valores `inf` (erro de dado, não outlier).
- **Limitação:** o IQR gera limites sem sentido em variáveis de contagem (limite inferior negativo em `n_accounts`, por exemplo). Nelas, a análise deve ser por frequência.

### 2.7 Correlação (seção 5)
- **O que faz:** matriz de Pearson, heatmap e lista de pares com correlação absoluta maior ou igual a 0,70.
- **Por quê:** variáveis muito correlacionadas carregam a mesma informação e causam instabilidade (multicolinearidade) em modelos lineares.
- **Achado:** só um par passa do corte, `n_accounts` e `n_issues` (0,9992), praticamente a mesma variável.
- **Limitação:** Pearson só captura relação linear e só se aplica às numéricas.

### 2.8 Variável-alvo (seção 6)
| Valor de `target_default` | Frequência | % |
|---|---|---|
| False (adimplente) | 35.080 | 77,96% |
| True (inadimplente) | 6.661 | 14,80% |
| NaN (sem rótulo) | 3.259 | 7,24% |

- **Desbalanceamento:** entre os rotulados, cerca de 16% são inadimplentes. Isso importa para a escolha de métricas (acurácia engana) e para técnicas de balanceamento.
- **Linhas sem rótulo:** 7,2% da base não tem alvo e não servem para treinar um modelo supervisionado.
- **`target_fraud`:** o crosstab (célula 57) é calculado só sobre as linhas em que há fraude registrada. Como a coluna é 96,6% nula, é provável que o NaN signifique "sem fraude", o que precisa ser confirmado. Além disso, o tipo de fraude só é conhecido depois do fato e não deve entrar como preditor (vazamento de dados).

### 2.9 Limpeza e transformação (seção 7)
O que foi feito:

1. **Base de trabalho** (`df_tratado`): cópia do original, para nunca alterar o bruto.
2. **Padronização dos nomes de colunas:** minúsculas, sem espaços.
3. **Limpeza de texto:** remoção de espaços e conversão de `''`, `'NA'`, `'N/A'`, `'NULL'`, `'None'` em nulo verdadeiro.
4. **Duplicatas exatas:** verificadas e removidas (nenhuma encontrada).
5. **Diagnóstico de variáveis codificadas:** `score_1`, `score_2`, `reason`, `state`, `zip`, `channel`, `job_name` e `real_state` estão anonimizadas por hash (ex.: `DGCQep2A...==`). Seu significado original é desconhecido, mas isso não as torna inúteis: ainda servem como categorias.
6. **Alta cardinalidade:** `reason` (14.874 valores) e `job_name` (32.265) têm categorias demais para one-hot encoding direto.
7. **Base para modelagem** (`df_modelagem`): remove `ids`, `profile_phone_number` (identificadores sem valor preditivo e potencialmente pessoais) e `channel` (constante).
8. **Conversão de data:** `application_time_applied` para `datetime`.
9. **Variáveis binárias:** identificação das colunas com exatamente 2 valores.
10. **Relatório de qualidade** e **comparação original × tratado.**
11. **Exportação** para `acquisition_train_tratado.csv`.

---

## 3. Limitações da Etapa 1

É importante ser honesto sobre o que esta etapa entrega: **ela diagnostica muito e trata pouco.** Pontos a corrigir antes de seguir:

| # | Problema | Efeito |
|---|---|---|
| 1 | Célula 3 aponta para `credit_risk_dataset.csv` (outra base) e depende do Colab | Confusão e quebra fora do Colab |
| 2 | `reported_income` tem `inf` e nada foi feito | Quebra escalonamento e modelos |
| 3 | `df_modelagem` é criado, mas o que se exporta é `df_tratado` | O CSV tratado ainda contém `ids`, telefone e `channel` |
| 4 | Conversão da data ocorre depois da cópia para `df_modelagem` | A base de modelagem fica com a data em texto |
| 5 | Nulos subiram de 219.437 para 222.554 após a limpeza de texto | Pode haver valores legítimos sendo apagados; não foi investigado |
| 6 | `target_default` com 3.259 nulos sem decisão | Linhas inúteis para treino |
| 7 | Ausências estruturais sem tratamento | Informação perdida |
| 8 | Constante `external_data_provider_credit_checks_last_2_year` mantida | Coluna sem informação |
| 9 | Par `n_accounts` / `n_issues` redundante, sem ação | Multicolinearidade |
| 10 | Outliers só diagnosticados | Distorção em modelos sensíveis |
| 11 | Heatmap duplicado (células 48 e 49) | Poluição visual |

---

## 4. Procedimentos para a próxima etapa (Etapa 2)

### 4.1 Correções pendentes da Etapa 1 (fazer primeiro)
1. Remover o trecho do Drive e do `credit_risk_dataset.csv` da célula 3.
2. Converter `inf` em NaN em `reported_income` (e verificar as demais colunas com `np.isinf`).
3. Investigar os ~3.100 nulos novos: comparar nulos por coluna entre `df_original` e `df_tratado`.
4. Converter `application_time_applied` **antes** de criar `df_modelagem`.
5. Exportar `df_modelagem`, e não `df_tratado`.

### 4.2 Definição do alvo
1. Separar as linhas com `target_default` nulo (3.259) do conjunto de treino. Podem ser guardadas à parte, como "não rotuladas".
2. Converter o alvo para 0/1.
3. Confirmar com a fonte o significado de NaN em `target_fraud`. Se for "sem fraude", tratá-lo como categoria própria; **não usar como preditor.**

### 4.3 Divisão treino/teste (antes de qualquer imputação ou escalonamento)
1. Dividir com estratificação pelo alvo (`stratify=y`), mantendo a proporção de inadimplentes (cerca de 16%).
2. Se a data tiver relevância temporal, preferir divisão por tempo (treino no passado, teste no futuro), que reflete melhor o uso real.
3. **Todos os parâmetros de imputação, escalonamento e encoding devem ser ajustados só no treino** e apenas aplicados no teste. Fazer isso na base inteira vaza informação do teste para o treino e infla as métricas.

### 4.4 Tratamento de valores ausentes
| Tipo de ausência | Variáveis | Estratégia sugerida |
|---|---|---|
| Estrutural (provável "não se aplica") | `last_amount_borrowed`, `last_borrowed_in_months`, `ok_since` | Criar flag `*_ausente` e preencher com 0 ou valor neutro, após confirmar a hipótese |
| Bloco de 562 linhas | `score_1/2/3`, `state`, `zip`, `channel`, `real_state`, `risk_rate` | Criar flag única `sem_dados_bureau` e imputar ou tratar como categoria "desconhecido" |
| Moderada/alta | `credit_limit`, `external_data_provider_credit_checks_last_year`, `n_issues` | Mediana (numéricas) com flag de ausência, avaliando se o nulo é informativo |
| Muito alta e sem informação | `target_fraud` | Fora do modelo |

A própria ausência pode ser informativa em crédito (quem não informa renda pode ter perfil de risco diferente), por isso as flags valem a pena.

### 4.5 Seleção e remoção de variáveis
1. Remover constantes: `channel`, `external_data_provider_credit_checks_last_2_year`.
2. Remover identificadores: `ids`, `profile_phone_number`.
3. Entre `n_accounts` e `n_issues`, manter só uma (ou ambas com cautela em modelos de árvore).
4. Avaliar `n_defaulted_loans` (quase constante), mas sem descartar de imediato, pois eventos raros podem ser muito preditivos.
5. Verificar outras possíveis redundâncias além do corte de 0,70, incluindo variáveis categóricas (V de Cramér).

### 4.6 Outliers e assimetria
1. `income`, `credit_limit`: aplicar `log1p` ou winsorização (por exemplo nos percentis 1 e 99), sem remover linhas.
2. Variáveis de contagem: não usar IQR; analisar por frequência e agrupar caudas (ex.: "3 ou mais").
3. Modelos de árvore (random forest, gradient boosting) são pouco sensíveis a outliers; modelos lineares exigem esse tratamento.

### 4.7 Variáveis categóricas
| Variável | Cardinalidade | Estratégia |
|---|---|---|
| `score_1`, `real_state` | Baixa (7 e 6) | One-hot encoding |
| `score_2`, `state` | Média (35 e 50) | One-hot com agrupamento de raras, ou target encoding |
| `zip` | 823 | Target encoding ou frequência, com validação cruzada |
| `reason`, `job_name` | Muito alta | Frequência, target encoding ou descartar; avaliar o ganho antes de manter |

Como os valores são hashes, o target encoding precisa ser feito **dentro de cada fold** para evitar vazamento.

### 4.8 Engenharia de variáveis
1. Extrair mês, dia da semana e hora de `application_time_applied`.
2. Razões entre variáveis, por exemplo `last_amount_borrowed / income` e `credit_limit / income`.
3. Diferença entre `income` e `reported_income` (indicador de inconsistência declarada).

### 4.9 Desbalanceamento e métricas
1. **Não usar acurácia** como métrica principal (prever sempre "adimplente" já daria cerca de 84%).
2. Usar AUC-ROC, AUC-PR, KS e recall da classe inadimplente.
3. Testar `class_weight='balanced'` antes de recorrer a SMOTE (e aplicar SMOTE só no treino).
4. Definir o ponto de corte pelo custo de negócio: aprovar um mau pagador custa mais do que negar um bom pagador, e essa assimetria deve guiar o threshold.

### 4.10 Modelagem
1. Baseline: regressão logística (interpretável, padrão em crédito).
2. Comparação: random forest e gradient boosting (XGBoost, LightGBM).
3. Validação cruzada estratificada e relatório com métricas por classe.
4. Interpretabilidade: coeficientes, importância de variáveis e SHAP, relevantes para justificar decisões de crédito.

### 4.11 Boas práticas de entrega
1. Usar `Pipeline` do scikit-learn com `ColumnTransformer`, para que todo o pré-processamento fique encapsulado e reprodutível.
2. Fixar `random_state`.
3. Remover o `warnings.filterwarnings('ignore')` global, que pode esconder avisos importantes.
4. Documentar cada decisão de tratamento (o quê, por quê e o impacto), como neste relatório.

---

## 5. Resumo executivo

- A base tem **45.000 registros únicos**, sem duplicatas.
- O principal problema é a **ausência de dados**, com várias colunas acima de 50%; boa parte parece estrutural e deve virar informação (flags), não ser apenas imputada.
- A variável-alvo está **desbalanceada** (cerca de 16% de inadimplentes entre os rotulados) e **7,2% das linhas não têm rótulo.**
- Muitas variáveis categóricas estão **anonimizadas por hash**, sendo úteis como categorias, mas não interpretáveis.
- A Etapa 1 **diagnosticou** bem, mas **tratou pouco**: os principais problemas (`inf`, alvo nulo, ausências, constantes, outliers) seguem abertos e são a base de trabalho da Etapa 2.
- A prioridade da Etapa 2: corrigir as pendências, dividir treino/teste **antes** de imputar ou escalonar, e escolher métricas adequadas ao desbalanceamento.
