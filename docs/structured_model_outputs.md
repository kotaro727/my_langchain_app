# 構造化されたモデル出力 (Structured model outputs)

モデルからのテキスト応答が、あなた（開発者）が定義したJSONスキーマに確実に準拠するようにします。

JSONは、アプリケーション間でデータをやり取りするために世界中で最も広く使われているフォーマットの一つです。

**Structured Outputs（構造化出力）** は、モデルが常にあなたが提供したJSONスキーマに従ったレスポンスを生成することを保証する機能です。これにより、モデルが必須のキーを省略したり、無効な列挙型（enum）の値を幻覚（ハルシネーション）として出力したりする心配がなくなります。

Structured Outputsの主な利点は以下の通りです:

*   **信頼性の高い型安全**: フォーマットが間違ったレスポンスを検証したり、再実行（リトライ）したりする必要がなくなります。
*   **明示的な拒絶検知**: 安全性に基づくモデルからの「回答拒否（Refusals）」を、プログラムで検知できるようになりました。
*   **シンプルなプロンプト**: 一貫した出力形式を得るために、プロンプトで語気を強めて厳格に指示を出す必要がなくなります。

REST APIでのJSONスキーマのサポートに加えて、PythonおよびJavaScript用のOpenAI SDKを使用すると、それぞれPydanticとZodを使ってオブジェクトスキーマを簡単に定義できます。以下は、コード内で定義されたスキーマに従って、構造化されていないテキストから情報を抽出する方法の例です。

## 構造化されたレスポンスの取得（Pythonコード例）

```python
from openai import OpenAI
from pydantic import BaseModel

client = OpenAI()

class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: list[str]

response = client.responses.parse(
    model="gpt-4o-2024-08-06",
    input=[
        {"role": "system", "content": "Extract the event information."},
        {
            "role": "user",
            "content": "Alice and Bob are going to a science fair on Friday.",
        },
    ],
    text_format=CalendarEvent, # ← Pydanticモデルを指定
)

event = response.output_parsed
```

## サポートされているモデル

Structured Outputsは、GPT-4oを皮切りとする最新の大規模言語モデルで利用可能です。`gpt-4-turbo` 以前の古いモデルでは、代わりに「JSONモード」を使用する場合があります。

## Function Calling と text.format の使い分け

OpenAI APIでは、Structured Outputsは以下の2つの形式で提供されています：

1.  Function Calling（関数呼び出し）を使用する場合
2.  `json_schema` response format を使用する場合

**Function Calling**は、モデルとアプリケーションの機能（システム）を橋渡しするアプリを構築する際に役立ちます。
例えば、ユーザーの注文を支援するAIアシスタントを構築するために、モデルに「データベースにクエリを送信する関数」や「UIを操作する関数」へのアクセス権を与えたい場合などです。

対照的に、**`response_format` 経由でのStructured Outputs**は、モデルが自力でツールを呼び出す時ではなく、「モデルがユーザーに返答する際」に特定の構造化スキーマを指定したい場合により適しています。
例えば、数学の家庭教師アプリを構築している場合、アシスタントからの返答を特定のJSONスキーマの形式で受け取り、UI側でモデルの出力の「各部分」を別々のデザインで表示したい場合などに使います。

**簡単に言えば:**
*   モデルをシステム内のツール、関数、データなどに接続している場合は、**Function Calling**を使用すべきです。
*   モデルがユーザーに返答する際に、その出力構造を整理して受け取りたい場合は、**text.format（response_format）**を使用すべきです。

（※このガイドの以降の部分では、Responses APIの非Function Callingのユースケースに焦点を当てます。）

## Structured Outputs vs JSON モード

Structured Outputsは「JSONモード」の進化版です。どちらも有効なJSONが出力されることは保証しますが、**特定のスキーマに準拠することを保証するのはStructured Outputsだけ**です。どちらのモードも主要なAPI（Responses, Chat Completions, Assistants, Fine-tuning, Batch API）でサポートされています。

可能であれば、**常にJSONモードの代わりにStructured Outputsを使用することを推奨**します。

ただし、`response_format: {type: "json_schema", ...}` によるStructured Outputsは、`gpt-4o-mini`, `gpt-4o-mini-2024-07-18`, `gpt-4o-2024-08-06` 以降のモデルスナップショットでのみサポートされています。

| 特徴 | Structured Outputs | JSON モード |
| :--- | :--- | :--- |
| 有効なJSONを出力するか | はい | はい |
| **スキーマに準拠するか** | **はい**（サポートされているスキーマ参照） | **いいえ** |
| 互換モデル | `gpt-4o-mini`, `gpt-4o-2024-08-06` およびそれ以降 | `gpt-3.5-turbo`, `gpt-4-*`, `gpt-4o-*` モデル |
| 有効化の設定 | `format: { type: "json_schema", "strict": true, "schema": ... }` | `format: { type: "json_object" }` |

## ユースケースの例

*   思考プロセス（Chain of thought）の出力
*   構造化データ抽出
*   UI生成
*   モデレーション

### 思考プロセス (Chain of thought)
ユーザーを解決策に導くために、モデルに構造化されたステップバイステップの方法で回答を出力するように依頼できます。

## Refusals（モデルによる回答拒否）の扱い

ユーザーが生成した入力に対しStructured Outputsを使用する場合、OpenAIモデルは安全上の理由からリクエストの実行を拒否することがあります。拒否された場合、当然ながらあなたが提供したスキーマには従わないため、APIレスポンスには新たに `refusal` というフィールドが含まれ、モデルがリクエストを拒否したことを示します。

レスポンスに `refusal` プロパティが含まれている場合、その拒否メッセージをUIに表示するか、コード内に条件分岐を追加して拒否時のエラーハンドリングを適切に行う必要があります。

```python
# 拒否された場合の処理例
if math_reasoning.refusal:
    print(math_reasoning.refusal) # 拒否メッセージを表示
else:
    print(math_reasoning.parsed)  # パースされた正常なデータを使用
```

## ヒントとベストプラクティス

**ユーザー入力の処理**
ユーザー入力を扱う場合、その入力が有効なレスポンス（スキーマ通り）に落とし込めない状況をどう処理するか、プロンプトに指示を含めるようにしてください。モデルは常に提供されたスキーマに従おうとするため、入力がスキーマと完全に無関係な場合、**幻覚（ハルシネーション）を引き起こす可能性**があります。
このような場合は、「タスクと互換性がないことを検知したら、空のパラメータを返す、あるいは特定の文章を返す」といった指示をプロンプトに入れると良いでしょう。

**間違いの処理**
Structured Outputsであっても、中身のロジックに間違いが含まれることがあります。間違いが見られる場合は、指示を調整する、システム指示に例（Few-shot）を追加する、またはタスクをより単純なサブタスクに分割してみてください。

**JSONスキーマとコードの型の乖離を防ぐ**
プログラミング言語での型定義とJSON Schemaが食い違うのを防ぐため、**ネイティブなPydanticやZodといったSDKサポート機能（型推論機能）の使用を強くお勧めします**。自前でJSON Schemaを直接書く場合は、コードの型定義との間にズレが生じないようCI（継続的インテグレーション）を組むなどの対策が必要です。

**ストリーミング**
レスポンス全体が完了するのを待つことなく、ストリーミングを使用して処理データを順次受け取ることも可能です。SDKを利用してストリーミングを実装することを推奨します。

## サポートされているスキーマの制約

Structured Outputsは、JSON Schema言語のサブセット（一部）のみをサポートしています。

**サポートされている型:**
`String` / `Number` / `Boolean` / `Integer` / `Object` / `Array` / `Enum` / `anyOf`

**文字列 (String) の制約:**
`pattern` (正規表現) や `format` (date, time, email, uuid等) が使用可能です。

**数値 (Number/Integer) の制約:**
`multipleOf`, `maximum`, `minimum`, `exclusiveMaximum`, `exclusiveMinimum` が指定可能です。

**配列 (Array) の制約:**
`minItems`, `maxItems` 指定が可能です。

### ⚠️ 重要：スキーマ設計時のルール・注意点

1.  **ルートオブジェクトは `anyOf` ではなく、単一の `object` でなければならない:**
    ルートレベル（一番外側）のスキーマは、必ず `type: "object"` である必要があります。Zodなどで判別可能なユニオン型（Discriminated Union）をトップレベルで使うとエラーになります。
2.  **すべてのフィールドは必須（`required`）でなければならない:**
    Structured Outputsを使用するには、全フィールドが提供されるよう定義する必要があります。もし「任意（Optional）」のパラメータをシミュレートしたい場合は、型を `["string", "null"]` のように Null とのUnion（anyOf）で定義してください。
3.  **オブジェクトのサイズと深さの制限:**
    合計5,000個までのプロパティ、最大10階層のネストまでサポートされています。
4.  **文字列とEnumサイズ制限:**
    すべてのキー名やEnum値の文字列の合計長は120,000文字を超えてはいけません。
5.  **`additionalProperties: false` は必須設定:**
    指定以外の予期せぬキーや値が生成されるのを防ぐため、すべてのオブジェクトにおいて `additionalProperties: false` （追加プロパティの許可を無効化）を明示的にセットしなければなりません。
6.  **出力順序の保証:**
    モデルは、あなたがスキーマ内で定義したキーの「順番通り」にデータを出力します。

---

## JSONモードについて（補足）
JSONモードはStructured Outputsの基本的なバージョンです。モデルの出力が有効なJSONであることを保証しますが、**特定のスキーマに一致することまでは保証しません**。可能であればStructured Outputsを使うことを推奨しますが、もし古いモデル等でJSONモードを使う場合は、以下の点に注意してください。

*   プロンプトのどこか（システムメッセージなど）で**「JSONで出力せよ」という文言を必ず明記**しなければなりません。明記しないと、無限に空白を生成し続けるなどのエラーの原因となります。
*   スキーマへの準拠は保証されないため、コード側で必ずバリデーション（Pydanticなどの検証ライブラリ）を行い、必要に応じてリトライ処理を実装する必要があります。
