# Privacy

Mlpipelinecheck 0.2 is a local library.

- Audits run in the calling process. The package does not open network connections and does not contain a telemetry client.
- Reports default to aggregates: row counts, missing rates, class proportions, means of numeric columns, and finding texts.
- Row fingerprints and group identifiers are counted. The hash and the group value are not written into the report. Prediction audits that receive `groups` or `timestamps` store counts, not the values.
- Raw feature rows and raw target sequences are not included. `privacy_include_examples=True` is rejected because that option is not implemented.
- Column names, class labels, and configuration values do appear, because the finding would not be readable without them. Those strings are HTML-escaped when a report is rendered.
- HTML is a single file with embedded CSS. It does not load scripts, fonts, or images from the network.
- CSV reading uses the path and encoding you pass to the CLI. The parser does not execute notebook cells, unpickle models, or run code stored in a dataset.
- Do not point the CLI at a file you are not willing to read locally. The tool reads the whole CSV into pandas.

If a report path is shared, treat column names and class labels as data. They can still be sensitive even when individual rows are absent.
