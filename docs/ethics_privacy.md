# Ethics, privacy, and limitations

## Consent

Face recognition is optional. Registration requires an explicit `consent=true` field. A real deployment must show a clear notice, provide a non-biometric alternative, record the purpose and retention period, and support deletion requests.

## Data minimization

The API processes uploaded images in memory and does not save them. Face registration stores a compact numerical embedding. Review and chat logs store one-way hashes rather than raw text.

## Bias and accuracy

Face recognition accuracy can differ across demographic groups, lighting, cameras, pose, and image quality. The included OpenCV baseline is educational and must not be used for consequential decisions. Evaluate false-match and false-non-match rates on a consented, representative test set.

## Human oversight

An automated result should never be the sole reason to deny service, change a price, accuse a shopper, or apply a penalty. Unknown or low-confidence results must be treated as inconclusive.

## Security

The included API-key check is a production-style demonstration, not complete identity and access management. A real system should use TLS, rotated secrets, user roles, audit logs, encryption at rest, rate limits, and a secrets manager.

## Model limitations

The bundled product classifier is a synthetic H5 centroid baseline for immediate execution. Replace it with the MobileNetV2 training path before reporting accuracy. The NLP datasets are also small demonstrations and should be replaced or expanded for final evaluation.
