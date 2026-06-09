use axum::{
    body::Body,
    http::{header, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
};
use include_dir::{include_dir, Dir};

static GUI_DIR: Dir<'_> = include_dir!("$CARGO_MANIFEST_DIR/../gui/out");

pub async fn serve_gui(uri_path: &str) -> Response {
    let rel = uri_path
        .strip_prefix("/gui/")
        .or_else(|| uri_path.strip_prefix("/gui"))
        .unwrap_or(uri_path)
        .trim_start_matches('/');

    let owned: Vec<String> = if rel.is_empty() {
        vec!["index.html".into()]
    } else {
        vec![
            rel.to_string(),
            format!("{rel}.html"),
            format!("{rel}/index.html"),
            "index.html".to_string(),
        ]
    };

    for candidate in &owned {
        if let Some(file) = GUI_DIR.get_file(candidate.as_str()) {
            let mime = mime_guess::from_path(candidate)
                .first_or_octet_stream()
                .to_string();
            return Response::builder()
                .status(StatusCode::OK)
                .header(
                    header::CONTENT_TYPE,
                    HeaderValue::from_str(&mime).unwrap_or_else(|_| {
                        HeaderValue::from_static("application/octet-stream")
                    }),
                )
                .body(Body::from(file.contents()))
                .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response());
        }
    }

    (StatusCode::NOT_FOUND, "GUI file not found").into_response()
}
