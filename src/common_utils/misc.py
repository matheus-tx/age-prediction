from jinjasql import JinjaSq
from omegaconf import DictConfig
from typing import Any


def make_statement(query_params: dict[str, Any],
                   raw_statement_path: str) -> str:
    """Make statement

    Parameters
    ----------
    `dates: dict[str, str]`
        Dates to query. Must have keys `first_earlier_query_date`,
        `last_earlier_query_date`, `first_later_query_date`,
        `last_later_query_date`.

    `raw_statement_path: str`
        Path to raw statement.

    Returns
    -------
    `statement: str`
        Prepared statement.

    Raises
    ------
    `ValueError`
        If dates are not ordered in ascending order.
    """
    raw_statement: str = open(raw_statement_path, "r").read()
    raw_statement, bind_params = JinjaSql(
        param_style="pyformat"
    ).prepare_query(source=raw_statement, data=query_params)
    statement: str = raw_statement % bind_params
    statement = statement.replace('[', '').replace(']', '').replace("'", "")
    print(statement)

    return statement


def _check(config: DictConfig) -> None:
    """Check

    Check connection configurations.

    Configurations must have keys `drivername`, `user`, `password`, `host` and
    `port`. It must also have either `service_name` or `database` keys. If
    these criteria are not met, an error is raised.

    COnnection by be made normally or via SSH. If SSH is to be used,
    configurations must have key `ssh`.

    Parameters
    ----------
    `config: DictConfig`
        Object with configurations. It must have the keys `drivername`,
        `user`, `password`, `host`, `port`. Is must also have either
        `service_name` or `database` keys. It may also have key `ssh` when SSH
        is to be used. In this case, its value must have keys `host`,
        `username`, `password` and `port`.

    Raises
    ------
    `ValueError`
        If at least one required key is absent or if both optional keys are
        missing.
    """
    def check_required(required_keys: list[str],
                       keys: list[str],
                       using_ssh: bool) -> None:
        missing_keys: list[str] = list(
            filter(lambda x: x not in keys, required_keys)
        )
        if len(missing_keys) != 0:
            ssh_string: str = '.ssh' if using_ssh else ''
            msg: str = (
                f'Required keys for `config{ssh_string}` are '
                f'`{required_keys}`. However, keys `{missing_keys}` are '
                'missing.'
            )
            raise ValueError(msg)

    # Check if one of required keys is missing
    REQUIRED_KEYS: list[str] = ['drivername', 'user', 'password', 'host',
                                'port']
    keys: list[Any] = list(config.keys())
    check_required(required_keys=REQUIRED_KEYS, keys=keys, using_ssh=False)

    if 'ssh' in keys:
        # Check if one of SSH keys is missing
        SSH_REQUIRED_KEYS: list[str] = ['host', 'username', 'password',
                                        'port']
        ssh_keys: list[str] = list(config.ssh.keys())
        check_required(required_keys=SSH_REQUIRED_KEYS,
                       keys=ssh_keys,
                       using_ssh=True)

    # Check if both optional keys are missing
    OPTIONAL_KEYS = ['database', 'service_name']
    if not any([key in keys for key in OPTIONAL_KEYS]):
        msg = (
            'Either `"database"` or `"service_name"` keys are required in '
            '`config`. However, both are missing.'
        )
        raise ValueError(msg)
