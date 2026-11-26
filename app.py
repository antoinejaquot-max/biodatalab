










    st.subheader("Previous analyses")

    results = st.session_state.get(
        "analysis_results",
        [],
    )

    if results:

        for item in results:

            with st.expander(
                item["analysis"]
            ):

                st.json(
                    item["results"]
                )

    else:

        st.info(
            "No analyses have been recorded yet."
        )

# ============================================================
# EXPORT
# ============================================================

elif page == "Export":

    st.header("Export")

    st.subheader("Dataset")

    csv_data = df.to_csv(
        index=False,
    ).encode("utf-8")

    st.download_button(
        "Download dataset CSV",
        csv_data,
        file_name="biodatalab_dataset.csv",
        mime="text/csv",
    )

    excel_buffer = io.BytesIO()

    with pd.ExcelWriter(
        excel_buffer,
        engine="openpyxl",
    ) as writer:

        df.to_excel(
            writer,
            index=False,
            sheet_name="Dataset",
        )

    st.download_button(
        "Download dataset Excel",
        excel_buffer.getvalue(),
        file_name="biodatalab_dataset.xlsx",
        mime=(
            "application/vnd.openxmlformats-officedocument"
            ".spreadsheetml.sheet"
        ),
    )

    st.subheader("Analysis results")

    results = st.session_state.get(
        "analysis_results",
        [],
    )

    st.download_button(
        "Download analyses JSON",
        export_results(),
        file_name="biodatalab_results.json",
        mime="application/json",
    )

    st.subheader("Reset")

    if st.button(
        "Clear analysis history",
        type="secondary",
    ):

        st.session_state["analysis_results"] = []

        st.success(
            "Analysis history cleared."
        )

# ============================================================
# FOOTER
# ============================================================

st.sidebar.divider()

st.sidebar.caption(
    "BioDataLab — Open-source scientific analysis"
)





