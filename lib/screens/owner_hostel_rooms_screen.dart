import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:google_fonts/google_fonts.dart';

import '../theme.dart';
import '../widgets/responsive_shell.dart';
import '../widgets/video_background.dart';
import 'owner_step_facilities.dart';

class OwnerHostelRoomsScreen extends StatefulWidget {
  final String ownerEmail;
  final String ownerName;
  final String hostelName;
  final String hostelType;
  final String city;
  final String address;
  final double latitude;
  final double longitude;

  const OwnerHostelRoomsScreen({
    super.key,
    required this.ownerEmail,
    required this.ownerName,
    required this.hostelName,
    required this.hostelType,
    required this.city,
    required this.address,
    required this.latitude,
    required this.longitude,
  });

  @override
  State<OwnerHostelRoomsScreen> createState() =>
      _OwnerHostelRoomsScreenState();
}

class _OwnerHostelRoomsScreenState
    extends State<OwnerHostelRoomsScreen>
    with SingleTickerProviderStateMixin {
  final _floorsCtrl = TextEditingController();
  final _capacityCtrl = TextEditingController();
  final _roomsCtrl = TextEditingController();

  String? _error;

  final List<Map<String, dynamic>> _roomTypes = [
    {
      'type': 'Single',
      'icon': Icons.bed_rounded,
      'color': AppTheme.tealMint,
      'selected': true,
      'priceCtrl': TextEditingController(
        text: '15000',
      ),
    },
    {
      'type': 'Double',
      'icon': Icons.king_bed_rounded,
      'color': const Color(0xFF69A9FF),
      'selected': true,
      'priceCtrl': TextEditingController(
        text: '10000',
      ),
    },
    {
      'type': 'Triple',
      'icon': Icons.bedroom_parent_rounded,
      'color': const Color(0xFFB487FF),
      'selected': false,
      'priceCtrl': TextEditingController(
        text: '8000',
      ),
    },
    {
      'type': 'Dormitory',
      'icon': Icons.hotel_rounded,
      'color': const Color(0xFFFFB76A),
      'selected': false,
      'priceCtrl': TextEditingController(
        text: '5000',
      ),
    },
  ];

  late final AnimationController _anim;
  late final Animation<double> _fade;
  late final Animation<double> _slide;

  // ===========================================================================
  // LIFECYCLE
  // ===========================================================================

  @override
  void initState() {
    super.initState();

    _anim = AnimationController(
      vsync: this,
      duration: const Duration(
        milliseconds: 750,
      ),
    );

    _fade = CurvedAnimation(
      parent: _anim,
      curve: Curves.easeOut,
    );

    _slide = Tween<double>(
      begin: 30,
      end: 0,
    ).animate(
      CurvedAnimation(
        parent: _anim,
        curve: Curves.easeOutCubic,
      ),
    );

    _anim.forward();
  }

  @override
  void dispose() {
    _anim.dispose();

    _floorsCtrl.dispose();
    _capacityCtrl.dispose();
    _roomsCtrl.dispose();

    for (final room in _roomTypes) {
      final controller =
          room['priceCtrl'] as TextEditingController;

      controller.dispose();
    }

    super.dispose();
  }

  // ===========================================================================
  // NEXT
  // ===========================================================================

  void _next() {
    FocusScope.of(context).unfocus();

    final floors = int.tryParse(
      _floorsCtrl.text.trim(),
    );

    final capacity = int.tryParse(
      _capacityCtrl.text.trim(),
    );

    final rooms = int.tryParse(
      _roomsCtrl.text.trim(),
    );

    if (floors == null ||
        capacity == null ||
        rooms == null) {
      setState(() {
        _error =
            'Please enter valid numbers for floors, capacity and rooms.';
      });

      return;
    }

    if (floors < 1 ||
        capacity < 1 ||
        rooms < 1) {
      setState(() {
        _error =
            'Floors, capacity and rooms must be at least 1.';
      });

      return;
    }

    final selectedRooms = _roomTypes
        .where(
          (room) => room['selected'] == true,
        )
        .toList();

    if (selectedRooms.isEmpty) {
      setState(() {
        _error =
            'Please select at least one room type.';
      });

      return;
    }

    for (final room in selectedRooms) {
      final controller =
          room['priceCtrl'] as TextEditingController;

      final price = int.tryParse(
        controller.text.trim(),
      );

      if (price == null || price <= 0) {
        setState(() {
          _error =
              'Please enter a valid price for every selected room type.';
        });

        return;
      }
    }

    final roomTypes = selectedRooms.map((room) {
      final controller =
          room['priceCtrl'] as TextEditingController;

      return <String, dynamic>{
        'type': room['type'].toString(),
        'price': int.parse(controller.text.trim()),
      };
    }).toList();

    setState(() {
      _error = null;
    });

    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (_) =>
            OwnerStepFacilities(
          ownerEmail: widget.ownerEmail,
          ownerName: widget.ownerName,
          hostelName: widget.hostelName,
          hostelType: widget.hostelType,
          city: widget.city,
          address: widget.address,
          latitude: widget.latitude,
          longitude: widget.longitude,
          totalFloors: floors,
          totalRooms: rooms,
          totalCapacity: capacity,
          roomTypes: roomTypes,
        ),
      ),
    );
  }

  // ===========================================================================
  // BUILD
  // ===========================================================================

  @override
  Widget build(BuildContext context) {
    final keyboardHeight =
        MediaQuery.viewInsetsOf(context).bottom;

    return ResponsiveShell(
      child: Scaffold(
        resizeToAvoidBottomInset: true,
        backgroundColor: Colors.transparent,
        body: VideoBackground(
          assetPath:
              'assets/images/background.mp4',
          overlayOpacity: 0.28,
          cardMargin:
              const EdgeInsets.all(14),
          cardRadius: 32,
          child: SafeArea(
            child: LayoutBuilder(
              builder: (
                context,
                constraints,
              ) {
                return SingleChildScrollView(
                  keyboardDismissBehavior:
                      ScrollViewKeyboardDismissBehavior
                          .onDrag,
                  physics:
                      const BouncingScrollPhysics(),
                  padding: EdgeInsets.fromLTRB(
                    18,
                    12,
                    18,
                    22 + keyboardHeight,
                  ),
                  child: AnimatedBuilder(
                    animation: _anim,
                    builder: (
                      context,
                      child,
                    ) {
                      return Transform.translate(
                        offset: Offset(
                          0,
                          _slide.value,
                        ),
                        child: Opacity(
                          opacity: _fade.value,
                          child: child,
                        ),
                      );
                    },
                    child: Column(
                      crossAxisAlignment:
                          CrossAxisAlignment.start,
                      children: [
                        _buildHeader(),

                        const SizedBox(
                          height: 18,
                        ),

                        _buildMainCard(),
                      ],
                    ),
                  ),
                );
              },
            ),
          ),
        ),
      ),
    );
  }

  // ===========================================================================
  // HEADER
  // ===========================================================================

  Widget _buildHeader() {
    return Column(
      crossAxisAlignment:
          CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            InkWell(
              onTap: () =>
                  Navigator.maybePop(
                context,
              ),
              borderRadius:
                  BorderRadius.circular(
                50,
              ),
              child: Container(
                width: 42,
                height: 42,
                decoration:
                    BoxDecoration(
                  color:
                      const Color(
                    0xFF02383B,
                  ).withOpacity(
                    0.55,
                  ),
                  shape:
                      BoxShape.circle,
                  border:
                      Border.all(
                    color: Colors.white
                        .withOpacity(
                      0.22,
                    ),
                  ),
                ),
                child:
                    const Icon(
                  Icons
                      .arrow_back_rounded,
                  color:
                      Colors.white,
                  size: 20,
                ),
              ),
            ),

            const Spacer(),

            Container(
              padding:
                  const EdgeInsets
                      .symmetric(
                horizontal: 11,
                vertical: 6,
              ),
              decoration:
                  BoxDecoration(
                color:
                    const Color(
                  0xFF02383B,
                ).withOpacity(
                  0.58,
                ),
                borderRadius:
                    BorderRadius
                        .circular(
                  17,
                ),
                border:
                    Border.all(
                  color: AppTheme
                      .tealMint
                      .withOpacity(
                    0.24,
                  ),
                ),
              ),
              child:
                  Text(
                'Step 5 of 9',
                style:
                    GoogleFonts.dmSans(
                  color:
                      Colors.white,
                  fontSize:
                      9.5,
                  fontWeight:
                      FontWeight
                          .w700,
                ),
              ),
            ),
          ],
        ),

        const SizedBox(
          height: 16,
        ),

        _buildProgress(),

        const SizedBox(
          height: 16,
        ),

        Text(
          'Room & Bed\nDetails',
          style:
              GoogleFonts.playfairDisplay(
            fontSize: 36,
            height: 1.03,
            fontWeight:
                FontWeight.w700,
            color: Colors.white,
          ),
        ),

        const SizedBox(
          height: 12,
        ),

        Text(
          'Set your accommodation structure and room pricing.',
          style:
              GoogleFonts.dmSans(
            color:
                Colors.white.withOpacity(
              0.70,
            ),
            fontSize: 12.5,
            height: 1.4,
          ),
        ),
      ],
    );
  }

  Widget _buildProgress() {
    return Row(
      children:
          List.generate(
        9,
        (index) {
          final active =
              index < 5;

          return Expanded(
            child: Container(
              height:
                  active ? 4 : 3,
              margin:
                  EdgeInsets.only(
                right:
                    index < 8 ? 5 : 0,
              ),
              decoration:
                  BoxDecoration(
                color:
                    active
                        ? AppTheme.tealMint
                        : Colors.white
                            .withOpacity(
                            0.22,
                          ),
                borderRadius:
                    BorderRadius
                        .circular(
                  3,
                ),
              ),
            ),
          );
        },
      ),
    );
  }

  // ===========================================================================
  // MAIN GLASS CARD
  // ===========================================================================

  Widget _buildMainCard() {
    return Container(
      width: double.infinity,
      padding:
          const EdgeInsets.fromLTRB(
        18,
        16,
        18,
        20,
      ),
      decoration:
          BoxDecoration(
        gradient:
            LinearGradient(
          begin:
              Alignment.topLeft,
          end:
              Alignment.bottomRight,
          colors: [
            const Color(
              0xFF063F42,
            ).withOpacity(
              0.90,
            ),
            const Color(
              0xFF023336,
            ).withOpacity(
              0.90,
            ),
            const Color(
              0xFF021F22,
            ).withOpacity(
              0.92,
            ),
          ],
        ),
        borderRadius:
            BorderRadius.circular(
          28,
        ),
        border:
            Border.all(
          color:
              AppTheme.tealMint
                  .withOpacity(
            0.22,
          ),
        ),
        boxShadow: [
          BoxShadow(
            color:
                Colors.black
                    .withOpacity(
              0.28,
            ),
            blurRadius:
                26,
            offset:
                const Offset(
              0,
              11,
            ),
          ),
        ],
      ),
      child:
          Column(
        crossAxisAlignment:
            CrossAxisAlignment.start,
        children: [
          Center(
            child: Container(
              width: 38,
              height: 4,
              margin:
                  const EdgeInsets.only(
                bottom: 16,
              ),
              decoration:
                  BoxDecoration(
                color:
                    AppTheme.tealMint
                        .withOpacity(
                  0.30,
                ),
                borderRadius:
                    BorderRadius
                        .circular(
                  3,
                ),
              ),
            ),
          ),

          _buildCardHeading(),

          const SizedBox(
            height: 14,
          ),

          _buildHostelSummary(),

          const SizedBox(
            height: 16,
          ),

          _CounterField(
            label:
                'Total Floors',
            hint:
                'Enter floors',
            icon:
                Icons.layers_rounded,
            controller:
                _floorsCtrl,
            onIncrement:
                () {
              final value =
                  int.tryParse(
                        _floorsCtrl
                            .text,
                      ) ??
                      0;

              setState(() {
                _floorsCtrl.text =
                    (value + 1)
                        .toString();
              });
            },
            onDecrement:
                () {
              final value =
                  int.tryParse(
                        _floorsCtrl
                            .text,
                      ) ??
                      0;

              if (value > 1) {
                setState(() {
                  _floorsCtrl.text =
                      (value - 1)
                          .toString();
                });
              }
            },
          ),

          const SizedBox(
            height: 11,
          ),

          _CounterField(
            label:
                'Total Capacity',
            hint:
                'Enter capacity',
            icon:
                Icons.groups_rounded,
            controller:
                _capacityCtrl,
            onIncrement:
                () {
              final value =
                  int.tryParse(
                        _capacityCtrl
                            .text,
                      ) ??
                      0;

              setState(() {
                _capacityCtrl.text =
                    (value + 1)
                        .toString();
              });
            },
            onDecrement:
                () {
              final value =
                  int.tryParse(
                        _capacityCtrl
                            .text,
                      ) ??
                      0;

              if (value > 1) {
                setState(() {
                  _capacityCtrl.text =
                      (value - 1)
                          .toString();
                });
              }
            },
          ),

          const SizedBox(
            height: 11,
          ),

          _CounterField(
            label:
                'Total Rooms',
            hint:
                'Enter rooms',
            icon: Icons
                .meeting_room_rounded,
            controller:
                _roomsCtrl,
            onIncrement:
                () {
              final value =
                  int.tryParse(
                        _roomsCtrl.text,
                      ) ??
                      0;

              setState(() {
                _roomsCtrl.text =
                    (value + 1)
                        .toString();
              });
            },
            onDecrement:
                () {
              final value =
                  int.tryParse(
                        _roomsCtrl.text,
                      ) ??
                      0;

              if (value > 1) {
                setState(() {
                  _roomsCtrl.text =
                      (value - 1)
                          .toString();
                });
              }
            },
          ),

          const SizedBox(
            height: 20,
          ),

          _buildRoomTypeHeading(),

          const SizedBox(
            height: 10,
          ),

          ...List.generate(
            _roomTypes.length,
            (index) {
              return Padding(
                padding:
                    const EdgeInsets.only(
                  bottom: 9,
                ),
                child:
                    _buildRoomTypeCard(
                  index,
                ),
              );
            },
          ),

          if (_error != null) ...[
            const SizedBox(
              height: 4,
            ),
            _buildError(),
          ],

          const SizedBox(
            height: 16,
          ),

          _buildNextButton(),
        ],
      ),
    );
  }

  Widget _buildCardHeading() {
    return Row(
      children: [
        Container(
          width: 44,
          height: 44,
          decoration:
              BoxDecoration(
            color:
                AppTheme.teal
                    .withOpacity(
              0.22,
            ),
            borderRadius:
                BorderRadius.circular(
              13,
            ),
            border:
                Border.all(
              color: AppTheme
                  .tealMint
                  .withOpacity(
                0.17,
              ),
            ),
          ),
          child:
              const Icon(
            Icons.bed_rounded,
            color:
                AppTheme.tealMint,
            size: 21,
          ),
        ),

        const SizedBox(
          width: 11,
        ),

        Expanded(
          child:
              Column(
            crossAxisAlignment:
                CrossAxisAlignment
                    .start,
            children: [
              Text(
                'Accommodation Details',
                maxLines:
                    1,
                overflow:
                    TextOverflow
                        .ellipsis,
                style:
                    GoogleFonts.playfairDisplay(
                  color:
                      Colors.white,
                  fontSize:
                      19,
                  fontWeight:
                      FontWeight
                          .w700,
                ),
              ),

              const SizedBox(
                height: 2,
              ),

              Text(
                'Configure hostel capacity and pricing',
                maxLines:
                    1,
                overflow:
                    TextOverflow
                        .ellipsis,
                style:
                    GoogleFonts.dmSans(
                  color: Colors
                      .white
                      .withOpacity(
                    0.50,
                  ),
                  fontSize:
                      8.8,
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }

  // ===========================================================================
  // HOSTEL SUMMARY
  // ===========================================================================

  Widget _buildHostelSummary() {
    return Container(
      width: double.infinity,
      padding:
          const EdgeInsets.all(
        12,
      ),
      decoration:
          BoxDecoration(
        color:
            const Color(
          0xFF011F21,
        ).withOpacity(
          0.58,
        ),
        borderRadius:
            BorderRadius.circular(
          14,
        ),
        border:
            Border.all(
          color:
              Colors.white.withOpacity(
            0.10,
          ),
        ),
      ),
      child:
          Row(
        children: [
          Container(
            width: 38,
            height: 38,
            decoration:
                BoxDecoration(
              color:
                  AppTheme.teal
                      .withOpacity(
                0.18,
              ),
              borderRadius:
                  BorderRadius
                      .circular(
                11,
              ),
            ),
            child:
                const Icon(
              Icons.apartment_rounded,
              color:
                  AppTheme.tealMint,
              size: 18,
            ),
          ),

          const SizedBox(
            width: 10,
          ),

          Expanded(
            child:
                Column(
              crossAxisAlignment:
                  CrossAxisAlignment
                      .start,
              children: [
                Text(
                  widget.hostelName,
                  maxLines:
                      1,
                  overflow:
                      TextOverflow
                          .ellipsis,
                  style:
                      GoogleFonts.dmSans(
                    color:
                        Colors.white,
                    fontSize:
                        11,
                    fontWeight:
                        FontWeight
                            .w700,
                  ),
                ),

                const SizedBox(
                  height: 2,
                ),

                Text(
                  '${widget.city} · ${_formatHostelType(widget.hostelType)}',
                  maxLines:
                      1,
                  overflow:
                      TextOverflow
                          .ellipsis,
                  style:
                      GoogleFonts.dmSans(
                    color: Colors
                        .white
                        .withOpacity(
                      0.46,
                    ),
                    fontSize:
                        8.5,
                  ),
                ),
              ],
            ),
          ),

          const SizedBox(
            width: 7,
          ),

          Container(
            padding:
                const EdgeInsets
                    .symmetric(
              horizontal:
                  8,
              vertical:
                  5,
            ),
            decoration:
                BoxDecoration(
              color:
                  const Color(
                0xFF54C49A,
              ).withOpacity(
                0.14,
              ),
              borderRadius:
                  BorderRadius
                      .circular(
                9,
              ),
            ),
            child:
                Row(
              mainAxisSize:
                  MainAxisSize.min,
              children: [
                const Icon(
                  Icons
                      .location_on_rounded,
                  size: 11,
                  color:
                      Color(
                    0xFF78D9B5,
                  ),
                ),
                const SizedBox(
                  width: 3,
                ),
                Text(
                  'Pinned',
                  style:
                      GoogleFonts.dmSans(
                    color:
                        const Color(
                      0xFF78D9B5,
                    ),
                    fontSize:
                        7.8,
                    fontWeight:
                        FontWeight
                            .w800,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  String _formatHostelType(
    String value,
  ) {
    final lower =
        value.toLowerCase();

    if (lower == 'girls') {
      return 'Girls Hostel';
    }

    if (lower == 'boys') {
      return 'Boys Hostel';
    }

    return value;
  }

  // ===========================================================================
  // ROOM TYPES
  // ===========================================================================

  Widget _buildRoomTypeHeading() {
    return Row(
      children: [
        const Icon(
          Icons.category_rounded,
          color:
              AppTheme.tealMint,
          size: 15,
        ),

        const SizedBox(
          width: 7,
        ),

        Text(
          'Room Types & Pricing',
          style:
              GoogleFonts.dmSans(
            color:
                Colors.white,
            fontSize:
                12.5,
            fontWeight:
                FontWeight.w800,
          ),
        ),
      ],
    );
  }

  Widget _buildRoomTypeCard(
    int index,
  ) {
    final room =
        _roomTypes[index];

    final selected =
        room['selected'] == true;

    final color =
        room['color'] as Color;

    final controller =
        room['priceCtrl']
            as TextEditingController;

    return InkWell(
      onTap: () {
        setState(() {
          room['selected'] =
              !selected;

          _error = null;
        });
      },
      borderRadius:
          BorderRadius.circular(
        16,
      ),
      child:
          AnimatedContainer(
        duration:
            const Duration(
          milliseconds:
              180,
        ),
        padding:
            const EdgeInsets.all(
          11,
        ),
        decoration:
            BoxDecoration(
          color:
              selected
                  ? color.withOpacity(
                      0.10,
                    )
                  : const Color(
                      0xFF011F21,
                    ).withOpacity(
                      0.50,
                    ),
          borderRadius:
              BorderRadius.circular(
            16,
          ),
          border:
              Border.all(
            color:
                selected
                    ? color.withOpacity(
                        0.66,
                      )
                    : Colors.white
                        .withOpacity(
                        0.09,
                      ),
            width:
                selected
                    ? 1.1
                    : 0.8,
          ),
        ),
        child:
            LayoutBuilder(
          builder: (
            context,
            constraints,
          ) {
            return Row(
              children: [
                Container(
                  width: 40,
                  height: 40,
                  decoration:
                      BoxDecoration(
                    color: color
                        .withOpacity(
                      selected
                          ? 0.18
                          : 0.09,
                    ),
                    borderRadius:
                        BorderRadius
                            .circular(
                      12,
                    ),
                  ),
                  child:
                      Icon(
                    room['icon']
                        as IconData,
                    size: 19,
                    color:
                        selected
                            ? color
                            : Colors
                                .white
                                .withOpacity(
                                0.50,
                              ),
                  ),
                ),

                const SizedBox(
                  width: 10,
                ),

                Expanded(
                  child:
                      Column(
                    crossAxisAlignment:
                        CrossAxisAlignment
                            .start,
                    children: [
                      Text(
                        room['type']
                            .toString(),
                        maxLines:
                            1,
                        overflow:
                            TextOverflow
                                .ellipsis,
                        style:
                            GoogleFonts.dmSans(
                          color:
                              selected
                                  ? Colors
                                      .white
                                  : Colors
                                      .white
                                      .withOpacity(
                                      0.72,
                                    ),
                          fontSize:
                              11,
                          fontWeight:
                              FontWeight
                                  .w700,
                        ),
                      ),

                      const SizedBox(
                        height: 2,
                      ),

                      Text(
                        selected
                            ? 'Available'
                            : 'Disabled',
                        style:
                            GoogleFonts.dmSans(
                          color:
                              selected
                                  ? color
                                  : Colors
                                      .white
                                      .withOpacity(
                                      0.38,
                                    ),
                          fontSize:
                              7.7,
                        ),
                      ),
                    ],
                  ),
                ),

                const SizedBox(
                  width: 8,
                ),

                if (selected)
                  SizedBox(
                    width:
                        constraints
                                    .maxWidth <
                                300
                            ? 100
                            : 118,
                    child:
                        _priceField(
                      controller,
                      color,
                    ),
                  )
                else
                  Container(
                    padding:
                        const EdgeInsets
                            .symmetric(
                      horizontal:
                          9,
                      vertical:
                          5,
                    ),
                    decoration:
                        BoxDecoration(
                      color: Colors
                          .white
                          .withOpacity(
                        0.06,
                      ),
                      borderRadius:
                          BorderRadius
                              .circular(
                        9,
                      ),
                    ),
                    child:
                        Text(
                      'Off',
                      style:
                          GoogleFonts.dmSans(
                        color: Colors
                            .white
                            .withOpacity(
                          0.42,
                        ),
                        fontSize:
                            8,
                        fontWeight:
                            FontWeight
                                .w700,
                      ),
                    ),
                  ),
              ],
            );
          },
        ),
      ),
    );
  }

  Widget _priceField(
    TextEditingController controller,
    Color color,
  ) {
    return TextField(
      controller:
          controller,
      keyboardType:
          TextInputType.number,
      inputFormatters: [
        FilteringTextInputFormatter
            .digitsOnly,
        LengthLimitingTextInputFormatter(
          7,
        ),
      ],
      textAlign:
          TextAlign.center,
      style:
          GoogleFonts.dmSans(
        color:
            const Color(
          0xFF122F31,
        ),
        fontSize: 11,
        fontWeight:
            FontWeight.w800,
      ),
      decoration:
          InputDecoration(
        isDense: true,
        prefixText:
            'Rs ',
        prefixStyle:
            GoogleFonts.dmSans(
          color:
              const Color(
            0xFF607776,
          ),
          fontSize: 8,
          fontWeight:
              FontWeight.w600,
        ),
        filled: true,
        fillColor:
            Colors.white
                .withOpacity(
          0.94,
        ),
        contentPadding:
            const EdgeInsets
                .symmetric(
          horizontal:
              7,
          vertical:
              9,
        ),
        border:
            OutlineInputBorder(
          borderRadius:
              BorderRadius.circular(
            10,
          ),
          borderSide:
              BorderSide(
            color:
                color.withOpacity(
              0.30,
            ),
          ),
        ),
        enabledBorder:
            OutlineInputBorder(
          borderRadius:
              BorderRadius.circular(
            10,
          ),
          borderSide:
              BorderSide(
            color:
                color.withOpacity(
              0.28,
            ),
          ),
        ),
        focusedBorder:
            OutlineInputBorder(
          borderRadius:
              BorderRadius.circular(
            10,
          ),
          borderSide:
              BorderSide(
            color: color,
            width: 1.2,
          ),
        ),
      ),
      onTap: () {
        controller.selection =
            TextSelection(
          baseOffset: 0,
          extentOffset:
              controller
                  .text.length,
        );
      },
    );
  }

  // ===========================================================================
  // ERROR
  // ===========================================================================

  Widget _buildError() {
    return Container(
      width: double.infinity,
      padding:
          const EdgeInsets.symmetric(
        horizontal: 12,
        vertical: 10,
      ),
      decoration:
          BoxDecoration(
        color:
            const Color(
          0xFF7F2525,
        ).withOpacity(
          0.42,
        ),
        borderRadius:
            BorderRadius.circular(
          12,
        ),
        border:
            Border.all(
          color:
              const Color(
            0xFFFF7777,
          ).withOpacity(
            0.38,
          ),
        ),
      ),
      child:
          Row(
        crossAxisAlignment:
            CrossAxisAlignment.start,
        children: [
          const Icon(
            Icons
                .error_outline_rounded,
            color:
                Color(
              0xFFFF8A8A,
            ),
            size: 16,
          ),

          const SizedBox(
            width: 8,
          ),

          Expanded(
            child:
                Text(
              _error!,
              style:
                  GoogleFonts.dmSans(
                color:
                    const Color(
                  0xFFFFB5B5,
                ),
                fontSize:
                    10,
                height:
                    1.35,
              ),
            ),
          ),
        ],
      ),
    );
  }

  // ===========================================================================
  // NEXT BUTTON
  // ===========================================================================

  Widget _buildNextButton() {
    return SizedBox(
      width: double.infinity,
      height: 50,
      child:
          ElevatedButton(
        onPressed:
            _next,
        style:
            ElevatedButton.styleFrom(
          backgroundColor:
              AppTheme.teal,
          foregroundColor:
              Colors.white,
          elevation:
              0,
          shape:
              RoundedRectangleBorder(
            borderRadius:
                BorderRadius.circular(
              15,
            ),
          ),
        ),
        child:
            Row(
          mainAxisAlignment:
              MainAxisAlignment
                  .center,
          children: [
            Text(
              'Next: Facilities',
              style:
                  GoogleFonts.dmSans(
                fontSize:
                    13.5,
                fontWeight:
                    FontWeight
                        .w800,
              ),
            ),

            const SizedBox(
              width: 9,
            ),

            Container(
              width: 25,
              height: 25,
              decoration:
                  BoxDecoration(
                color:
                    Colors.white
                        .withOpacity(
                  0.15,
                ),
                shape:
                    BoxShape.circle,
              ),
              child:
                  const Icon(
                Icons
                    .arrow_forward_rounded,
                size: 14,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

// =============================================================================
// COUNTER FIELD
// =============================================================================

class _CounterField extends StatelessWidget {
  final String label;
  final String hint;
  final IconData icon;

  final TextEditingController
      controller;

  final VoidCallback
      onIncrement;

  final VoidCallback
      onDecrement;

  const _CounterField({
    required this.label,
    required this.hint,
    required this.icon,
    required this.controller,
    required this.onIncrement,
    required this.onDecrement,
  });

  @override
  Widget build(
    BuildContext context,
  ) {
    return Column(
      crossAxisAlignment:
          CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Icon(
              icon,
              size: 14,
              color: AppTheme
                  .tealMint
                  .withOpacity(
                0.85,
              ),
            ),

            const SizedBox(
              width: 6,
            ),

            Text(
              label,
              style:
                  GoogleFonts.dmSans(
                fontSize:
                    10.5,
                fontWeight:
                    FontWeight.w700,
                color:
                    Colors.white
                        .withOpacity(
                  0.72,
                ),
              ),
            ),
          ],
        ),

        const SizedBox(
          height: 6,
        ),

        Container(
          height: 49,
          decoration:
              BoxDecoration(
            color:
                Colors.white
                    .withOpacity(
              0.94,
            ),
            borderRadius:
                BorderRadius.circular(
              14,
            ),
            border:
                Border.all(
              color: AppTheme
                  .tealMint
                  .withOpacity(
                0.18,
              ),
            ),
          ),
          child:
              Row(
            children: [
              InkWell(
                onTap:
                    onDecrement,
                borderRadius:
                    const BorderRadius
                        .horizontal(
                  left:
                      Radius.circular(
                    14,
                  ),
                ),
                child:
                    Container(
                  width: 45,
                  height:
                      double.infinity,
                  decoration:
                      BoxDecoration(
                    color:
                        const Color(
                      0xFFE7F0EE,
                    ),
                    borderRadius:
                        const BorderRadius
                            .horizontal(
                      left:
                          Radius.circular(
                        14,
                      ),
                    ),
                  ),
                  child:
                      const Icon(
                    Icons
                        .remove_rounded,
                    size: 18,
                    color:
                        Color(
                      0xFF07595C,
                    ),
                  ),
                ),
              ),

              Expanded(
                child:
                    TextField(
                  controller:
                      controller,
                  textAlign:
                      TextAlign.center,
                  keyboardType:
                      TextInputType.number,
                  inputFormatters: [
                    FilteringTextInputFormatter
                        .digitsOnly,
                    LengthLimitingTextInputFormatter(
                      4,
                    ),
                  ],
                  style:
                      GoogleFonts.dmSans(
                    fontSize:
                        14,
                    fontWeight:
                        FontWeight
                            .w800,
                    color:
                        const Color(
                      0xFF183636,
                    ),
                  ),
                  decoration:
                      InputDecoration(
                    hintText:
                        hint,
                    hintStyle:
                        GoogleFonts.dmSans(
                      color:
                          const Color(
                        0xFF869695,
                      ),
                      fontSize:
                          10.5,
                    ),
                    border:
                        InputBorder.none,
                    contentPadding:
                        const EdgeInsets
                            .symmetric(
                      vertical:
                          13,
                    ),
                  ),
                ),
              ),

              InkWell(
                onTap:
                    onIncrement,
                borderRadius:
                    const BorderRadius
                        .horizontal(
                  right:
                      Radius.circular(
                    14,
                  ),
                ),
                child:
                    Container(
                  width: 45,
                  height:
                      double.infinity,
                  decoration:
                      const BoxDecoration(
                    color:
                        AppTheme.teal,
                    borderRadius:
                        BorderRadius
                            .horizontal(
                      right:
                          Radius.circular(
                        14,
                      ),
                    ),
                  ),
                  child:
                      const Icon(
                    Icons.add_rounded,
                    size: 18,
                    color:
                        Colors.white,
                  ),
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }
}