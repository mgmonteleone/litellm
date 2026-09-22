"""Utils for accessing credentials."""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

import litellm
from litellm.types.utils import CredentialItem

CREDENTIAL_BOUND_ENDPOINT_KEYS: Final = frozenset({"api_base", "aws_sts_endpoint", "aws_web_identity_token"})


class CredentialAccessor:
    @staticmethod
    def find_credential(credential_name: str) -> CredentialItem | None:
        return next(
            (credential for credential in litellm.credential_list if credential.credential_name == credential_name),
            None,
        )

    @staticmethod
    def get_credential_values(credential_name: str) -> dict:
        """Safe accessor for credentials."""

        credential: Final = CredentialAccessor.find_credential(credential_name)
        return {} if credential is None else credential.credential_values.copy()

    @staticmethod
    def endpoints_left_unset(credential_values: Mapping[str, object]) -> frozenset[str]:
        """A credential's secrets go only where the credential itself says: an endpoint it leaves unset falls back to
        the provider default rather than to whatever endpoint a caller or a saved row supplied next to its name."""
        return CREDENTIAL_BOUND_ENDPOINT_KEYS - credential_values.keys() if credential_values else frozenset()

    @staticmethod
    def endpoints_left_unset_by_name(credential_name: object) -> frozenset[str]:
        credential: Final = (
            CredentialAccessor.find_credential(credential_name) if isinstance(credential_name, str) else None
        )
        if credential is None:
            return frozenset()
        credential_values: Final[Mapping[object, object]] = credential.credential_values
        return CredentialAccessor.endpoints_left_unset(
            MappingProxyType({key: value for key, value in credential_values.items() if isinstance(key, str)})
        )

    @staticmethod
    def upsert_credentials(credentials: list[CredentialItem]):
        """Add a credential to the list of credentials."""

        credential_names: Final = [cred.credential_name for cred in litellm.credential_list]

        for credential in credentials:
            if credential.credential_name in credential_names:
                # Find and replace the existing credential in the list
                for i, existing_cred in enumerate(litellm.credential_list):
                    if existing_cred.credential_name == credential.credential_name:
                        litellm.credential_list[i] = credential
                        break
            else:
                litellm.credential_list.append(credential)
